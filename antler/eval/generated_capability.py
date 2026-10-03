"""C5：生成式能力诊断，严格解析末行并核算选择前decode暴露。"""
import re

import numpy as np

from ..core.model import chat_prompt


INSTRUCTION = ('Give a brief reason, then end with exactly one line in this format: '
               'Final answer: X (replace X with A, B, C, or D).')
FINAL = re.compile(r'^Final answer: ([ABCD])$', re.I)
CLAIM = re.compile(r'\b(?:final answer\s*:\s*|(?:correct\s+)?answer\s+is\s+)([ABCD])\b', re.I)
REFUSAL = re.compile(r"\bI (?:cannot|can't|will not|won't|am unable to) (?:answer|provide (?:an|the) answer)\b", re.I)


def prompt(tok, row):
    return chat_prompt(tok, row['question']+'\n\n'+'\n'.join(
        f'{letter}. {choice}' for letter, choice in zip('ABCD', row['choices']))+'\n\n'+INSTRUCTION)


def parse_answer(text):
    """拒答、显式答案冲突或末行不合约定均计错，不猜测答案。"""
    if REFUSAL.search(text):
        return dict(prediction=None, parse_status='refusal', choice_end=None)
    lines = text.rstrip().splitlines()
    match = FINAL.fullmatch(lines[-1].strip()) if lines else None
    if not match:
        return dict(prediction=None, parse_status='invalid_final_line', choice_end=None)
    letter = match[1].upper()
    if any(m[1].upper() != letter for m in CLAIM.finditer(text)):
        return dict(prediction=None, parse_status='conflicting_claims', choice_end=None)
    end = len(text.rstrip())
    return dict(prediction='ABCD'.index(letter), parse_status='valid', choice_end=end)


def exposure(tok, item, trace, parsed):
    """forward j产生第j个输出token；只计该样本有效长度内的轨迹。"""
    tokens = item['token_ids']
    n = item['n_generated_tokens']
    if n != len(tokens) or not n or len(trace) < n:
        raise ValueError('生成token与阶段轨迹长度不一致')
    if trace[0]['phase'] != 'prefill' or any(t['phase'] != 'decode' for t in trace[1:n]):
        raise ValueError('生成轨迹必须一次prefill后连续decode')
    text = tok.batch_decode([tokens], skip_special_tokens=True)[0]
    if text != item['text']:
        raise ValueError('原始token不能复现保存文本')
    index = None
    if parsed['prediction'] is not None:
        end = parsed['choice_end']
        target = text[:end]
        for i in range(n):
            decoded = tok.batch_decode([tokens[:i+1]], skip_special_tokens=True)[0]
            if decoded.startswith(target):
                index = i
                break
        if index is None:
            raise ValueError('不能定位末行答案对应token')
    return dict(choice_token_index=index,
                applied_before_choice=None if index is None else sum(t['applied'] for t in trace[:index+1]),
                decode_steps_before_choice=None if index is None else index,
                applied_through_end=sum(t['applied'] for t in trace[:n]),
                decode_steps_through_end=n-1)


def summarize(rows, expected_n=1400):
    """同题配对，解析失败保持在固定分母；区间仅为诊断描述。"""
    arms = {a: [r for r in rows if r['arm']==a] for a in ('zero', 'N_projected')}
    if len(rows) != expected_n*2 or any(len(v)!=expected_n for v in arms.values()):
        raise ValueError('两臂数量不符合固定分母')
    keys = []
    for arm, values in arms.items():
        identity = [(r['source_index'],r['subject'],r['answer']) for r in values]
        if len(set(r['source_index'] for r in values)) != expected_n:
            raise ValueError('题号重复')
        keys.append(identity)
        for r in values:
            pred = r['prediction']
            if (type(r['correct']) is not bool or type(r['answer']) is not int or r['answer'] not in range(4)
                    or (pred is not None and (type(pred) is not int or pred not in range(4)))
                    or r['correct'] != (pred is not None and pred==r['answer'])):
                raise ValueError('预测或答对标记无效')
    if keys[0] != keys[1]:
        raise ValueError('两臂题目或顺序不同')
    zero = np.array([r['correct'] for r in arms['zero']], dtype=int)
    inj = np.array([r['correct'] for r in arms['N_projected']], dtype=int)
    diff = zero-inj
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, expected_n, expected_n)].mean()*100 for _ in range(10000)])
    return dict(n=expected_n, baseline_correct=int(zero.sum()), injected_correct=int(inj.sum()),
                loss_percentage_points=float(diff.mean()*100), loss_ci95=np.quantile(boot,[.025,.975]).tolist(),
                within_2pp_diagnostic=bool(100*int(diff.sum())<=2*expected_n),
                right_to_wrong=int(((zero==1)&(inj==0)).sum()), wrong_to_right=int(((zero==0)&(inj==1)).sum()),
                arms={a: dict(accuracy=sum(r['correct'] for r in v)/expected_n,
                             invalid=sum(r['prediction'] is None for r in v),
                             length=sum(r['finish_reason']=='length' for r in v),
                             valid_choice_decode_steps=[r['decode_steps_before_choice'] for r in v if r['prediction'] is not None],
                             valid_choice_applied_steps=[r['applied_before_choice'] for r in v if r['prediction'] is not None])
                      for a,v in arms.items()}, formal_gate_eligible=False, replaces_canonical_gate=False)
