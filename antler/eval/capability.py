"""C5：固定学科分层样本，以选项字母logit评测MMLU。"""
from collections import defaultdict
import random

import torch

from ..core.hooks import Steer, no_steer
from ..core.model import chat_prompt, model_device


def stratified_sample(rows, total=1400, seed=0):
    """按学科近似均匀分配，固定随机种子与原始行号，不按成绩挑题。"""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if not row.get('subject') or len(row.get('choices', [])) != 4 or row.get('answer') not in range(4):
            raise ValueError('MMLU需要subject、四选项和0–3整数答案')
        groups[row['subject']].append({**row, 'source_index': index})
    if not groups or total < len(groups) or total > len(rows):
        raise ValueError('样本数必须覆盖全部学科且不超过题库')
    rng = random.Random(seed)
    subjects = sorted(groups)
    for subject in subjects:
        rng.shuffle(groups[subject])
    selected = []
    # 轮转抽样使学科配额相差至多一题；小科目耗尽后由其他科目补足。
    while len(selected) < total:
        for subject in subjects:
            if groups[subject] and len(selected) < total:
                selected.append(groups[subject].pop())
    return selected


@torch.no_grad()
def mmlu_accuracy(model, tok, rows, *, steer=None, batch_size=16):
    """只在A/B/C/D之间比较logit；提示无示例，结束在助手回答起点。"""
    if not rows or batch_size < 1:
        raise ValueError('题目非空且batch_size为正')
    answer_ids = [tok.encode(letter, add_special_tokens=False) for letter in 'ABCD']
    if any(len(ids) != 1 for ids in answer_ids) or len({ids[0] for ids in answer_ids}) != 4:
        raise ValueError('A/B/C/D必须对应四个不同的单token')
    letters = [ids[0] for ids in answer_ids]
    tok.padding_side = 'left'
    results = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset+batch_size]
        prompts = [chat_prompt(tok, row['question'] + '\n\n' + '\n'.join(
            f'{letter}. {choice}' for letter, choice in zip('ABCD', row['choices']))
            + '\n\nAnswer with a single letter: A, B, C, or D.') for row in batch]
        inputs = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(model_device(model))
        with Steer(model, steer) if steer else no_steer():
            logits = model(**inputs).logits[:, -1, letters].float()
        predictions = logits.argmax(dim=-1).tolist()
        for row, prediction in zip(batch, predictions):
            results.append({'source_index': row['source_index'], 'subject': row['subject'],
                            'answer': int(row['answer']), 'prediction': prediction,
                            'correct': prediction == int(row['answer'])})
    return {'accuracy': sum(r['correct'] for r in results)/len(results), 'n': len(results), 'samples': results}
