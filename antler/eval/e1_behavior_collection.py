"""C11：行为采集底层组件；调用方必须先完成具体注入条件的验证和登记。"""
import hashlib
import random

import torch

from ..core.generate import generate
from ..core.hooks import STEER_PHASES
from ..core.model import chat_prompt, get_layers
from ..memory.write import BASE_SYSTEM, MEMORY_TEMPLATE, NEUTRAL_MEMORIES
from ..traits.data import BIG_FIVE, load_trait
from .e1_behavior import summarize_behavior


def collect_answers(model, tok, *, experience, dim, pole, conditions, seeds,
                    max_new_tokens=320, temperature=1., on_record=None):
    """采集一个经历的无记忆、五中性、上下文及各注入臂，保留真实token。

    conditions为{name: {layer, vector, phase}}；只接受调用端已验证条件。
    此底层函数不能自行验证能力报告或批准条件，不提供可绕过门槛的CLI。
    on_record用于上层逐条落盘；调用方负责run_meta、哈希及一次性运行登记。
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('必须使用冻结且处于eval模式的模型')
    if not isinstance(experience, str) or not experience.strip():
        raise ValueError('经历不能为空')
    if dim not in BIG_FIVE or pole not in ('high', 'low'):
        raise ValueError('维度或极点非法')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('须显式指定互不重复的非负采样种子')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('最大生成长度须为正整数')
    if not isinstance(temperature, (int, float)) or not 0 <= temperature < float('inf'):
        raise ValueError('温度须为有限非负数')
    checked = {}
    for name, c in conditions.items():
        if not isinstance(name, str) or not name or name in ('none', 'neutral', 'context'):
            raise ValueError('注入臂名称非法或与对照冲突')
        layer, vector, phase = c['layer'], c['vector'], c['phase']
        if (type(layer) is not int or not 0 <= layer < len(get_layers(model))
                or phase not in STEER_PHASES or not isinstance(vector, torch.Tensor)
                or vector.shape != (model.config.hidden_size,) or not torch.isfinite(vector).all()
                or not vector.is_floating_point()):
            raise ValueError('注入条件的层、阶段或向量非法')
        # 复制，避免上层在采集中原地改动向量。
        vector = vector.detach().cpu().float().contiguous().clone()
        checked[name] = {'layer': layer, 'vector': vector, 'phase': phase,
                         'vector_sha256': hashlib.sha256(vector.numpy().tobytes()).hexdigest()}
    data = load_trait(BIG_FIVE[dim][0 if pole == 'high' else 1], 'eval')
    if len(data['questions']) != 20:
        raise ValueError('评测集必须满足固定20题切分')
    arms = [('none', None, BASE_SYSTEM),
            ('context', None, MEMORY_TEMPLATE.format(experience=experience))]
    arms += [('neutral', i, MEMORY_TEMPLATE.format(experience=m)) for i, m in enumerate(NEUTRAL_MEMORIES)]
    arms += [(name, i, MEMORY_TEMPLATE.format(experience=m))
             for name in checked for i, m in enumerate(NEUTRAL_MEMORIES)]
    records = []
    for name, memory_index, system in arms:
        c = checked.get(name)
        for sample_index, seed in enumerate(seeds):
            for question_index in range(10, 20):
                q = data['questions'][question_index]
                prompt = chat_prompt(tok, q, system=system)
                trace = []
                answer = generate(model, tok, [prompt],
                    steer={c['layer']: c['vector']} if c else None,
                    steer_phase=c['phase'] if c else 'all', max_new_tokens=max_new_tokens,
                    temperature=temperature, batch_size=1, seed=seed,
                    return_details=True, return_token_ids=True, phase_trace_out=trace)[0]
                row = {'arm': name, 'memory_index': memory_index, 'dim': dim, 'pole': pole,
                       'split': 'test', 'question_index': question_index, 'sample_index': sample_index,
                       'seed': seed, 'q': q, 'system': system, 'prompt': prompt, 'a': answer['text'],
                       'token_ids': answer['token_ids'], 'n_generated_tokens': answer['n_generated_tokens'],
                       'finish_reason': answer['finish_reason'], 'phase_trace': trace,
                       'condition': None if c is None else {k: c[k] for k in ('layer', 'phase', 'vector_sha256')}}
                records.append(row)
                if on_record is not None:
                    on_record(row)
    return records


def score_answers(judge, records, *, seed=0):
    """同维同极点全部实验臂合并乱序，仅向固定评委提供题目和答案。"""
    if not records:
        raise ValueError('不能评分空记录')
    dim, pole = records[0]['dim'], records[0]['pole']
    if dim not in BIG_FIVE or pole not in ('high', 'low'):
        raise ValueError('维度或极点非法')
    if any((r['dim'], r['pole'], r['split']) != (dim, pole, 'test') for r in records):
        raise ValueError('评分批次必须为同维同极点test记录')
    keys = [(r['arm'], r['memory_index'], r['question_index'], r['sample_index']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('采集身份重复')
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    qs, answers = [records[i]['q'] for i in order], [records[i]['a'] for i in order]
    prompt = load_trait(BIG_FIVE[dim][0 if pole == 'high' else 1], 'eval')['eval_prompt']
    scores = {'trait': judge.trait_scores(prompt, qs, answers),
              'coherence': judge.coherence_scores(qs, answers), 'task': judge.task_scores(qs, answers)}
    if any(len(v) != len(order) for v in scores.values()):
        raise ValueError('评委返回条数与请求不一致')
    result = [dict(r) for r in records]
    for j, original in enumerate(order):
        result[original].update({metric: values[j] for metric, values in scores.items()})
        result[original]['judge_order_index'] = j
    # None原样保存，由汇总组件拒绝；不自动重问或改分。
    return result


def summarize_scored_answers(records, *, intended_sign, seed=0):
    """把采集端身份映射到五记忆汇总，不以成功汇总代替条件验收。"""
    if not records:
        raise ValueError('不能汇总空记录')
    grouped = {}
    for row in records:
        grouped.setdefault((row['arm'], row['memory_index']), []).append(row)
    names = {r['arm'] for r in records} - {'none', 'context', 'neutral'}
    required = {('none', None), ('context', None)} | {('neutral', i) for i in range(5)}
    required |= {(name, i) for name in names for i in range(5)}
    if set(grouped) != required:
        raise ValueError('缺少对照或五记忆注入记录，或存在额外记忆身份')
    reference = records[0]
    if any((r['dim'], r['pole'], r['split']) !=
           (reference['dim'], reference['pole'], 'test') for r in records):
        raise ValueError('采集记录维度、极点或切分不一致')
    def report(name, memory):
        rows = grouped[name, memory]
        if name == 'none':
            expected_system = BASE_SYSTEM
        elif name == 'context':
            expected_system = rows[0]['system']
            prefix = MEMORY_TEMPLATE.split('{experience}')[0]
            if not expected_system.startswith(prefix) or len(expected_system) == len(prefix):
                raise ValueError('上下文提示缺少固定模板或经历')
        else:
            expected_system = MEMORY_TEMPLATE.format(experience=NEUTRAL_MEMORIES[memory])
        if any(r['system'] != expected_system for r in rows):
            raise ValueError('同一记忆身份的系统提示改变')
        return {'dim': reference['dim'], 'pole': reference['pole'], 'split': 'test', 'samples': rows}
    context = report('context', None)
    baselines = [report('neutral', i) for i in range(5)]
    reference_seeds = {(r['question_index'], r['sample_index']): r['seed'] for r in context['samples']}
    for r in records:
        if r['seed'] != reference_seeds.get((r['question_index'], r['sample_index'])):
            raise ValueError('各臂采样种子不配对')
        if r['arm'] in ('none', 'context', 'neutral') and r['condition'] is not None:
            raise ValueError('对照臂不能带注入条件')
    # 无记忆臂也经过完整评分身份检查，不把它代入五中性迁移比分母。
    no_memory = report('none', None)
    summarize_behavior(baselines, no_memory, baselines, intended_sign=intended_sign, seed=seed)
    result = {}
    for name in sorted(names):
        rows = [r for r in records if r['arm'] == name]
        condition = rows[0]['condition']
        if condition is None or any(r['condition'] != condition for r in rows):
            raise ValueError('同一注入臂的条件哈希、层或阶段改变')
        result[name] = summarize_behavior(baselines, context, [report(name, i) for i in range(5)],
                                          intended_sign=intended_sign, seed=seed)
        result[name]['condition'] = condition
    return {'arms': result, 'formal_e1_pass': False, 'behavior_validated': False}
