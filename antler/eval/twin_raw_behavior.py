"""C12：原幅度状态行为工作清单与五题开发诊断，不授予正式资格。"""
from __future__ import annotations

import math
import random

import numpy as np

from antler.eval.twin_context_gate import jobs_for
from antler.memory.twin_state import CELLS
from antler.memory.write import MEMORY_TEMPLATE, NEUTRAL_MEMORIES
from antler.traits.data import BIG_FIVE, load_trait


FORMATS = ('full_state', 'reconstructed_state')
METRICS = ('trait', 'coherence', 'task')
SCENARIO = 'twin04_ideas'
PROTOCOL = {'layer': 16, 'phase': 'all', 'factor': 1., 'seed': 0, 'dim': 'E',
            'split': 'calib', 'question_indices': list(range(5)), 'neutral_count': 5,
            'max_new_tokens': 768, 'temperature': 1., 'top_p': 1., 'batch_size': 8,
            'max_generated': 400, 'reused': 90, 'max_answers': 490, 'max_scores': 1470,
            'shuffle_seed': 0, 'bootstrap_seed': 0, 'bootstrap_samples': 10000, 'retry': False}


def expected_ids():
    return [f'a{a}_r{r}_{fmt}' for a, r in CELLS for fmt in FORMATS]


def build_jobs(materials, capability):
    """先保存全部固定对照，再按八候选原序收集合格者，不挑故事。"""
    if set(capability) != set(expected_ids()) or any(type(v.get('within_2pp')) is not bool for v in capability.values()):
        raise ValueError('能力结果必须包含完整八候选与明确布尔判定')
    originals = [j for j in jobs_for(materials) if j['dim'] == 'E' and j['seed'] == 0
                 and j['question_index'] < 5 and (j['arm'] == 'neutral' or
                 (j['arm'] == 'context' and j['scenario_id'] == SCENARIO))]
    if len(originals) != 90:
        raise ValueError('固定复用对照预算不符')
    jobs = [{**j, 'id': i, 'upstream_id': j['id'], 'vector_id': None, 'format': None} for i, j in enumerate(originals)]
    for a, r in CELLS:
        for fmt in FORMATS:
            vector_id = f'a{a}_r{r}_{fmt}'
            if not capability[vector_id]['within_2pp']:
                continue
            for memory, text in enumerate(NEUTRAL_MEMORIES):
                for pole in ('high', 'low'):
                    trait = load_trait(BIG_FIVE['E'][pole == 'low'], 'eval')
                    for q in range(5):
                        jobs.append({'id': len(jobs), 'dim': 'E', 'pole': pole, 'split': 'calib',
                                     'arm': 'injected', 'memory_index': memory,
                                     'episode_id': f'{SCENARIO}_a{a}_r{r}', 'scenario_id': SCENARIO,
                                     'action_sign': a, 'outcome_sign': r, 'seed': 0, 'sample_index': 0,
                                     'question_index': q, 'q': trait['questions'][q],
                                     'system': MEMORY_TEMPLATE.format(experience=text),
                                     'upstream_id': None, 'vector_id': vector_id, 'format': fmt})
    return jobs


def reused_answers(jobs, upstream):
    """只取原文及生成证据；旧分数、旧评分顺序不进入本轮。"""
    if len({r['id'] for r in upstream}) != len(upstream):
        raise ValueError('上游回答身份重复')
    indexed = {r['id']: r for r in upstream}
    result = []
    for job in jobs:
        if job['upstream_id'] is None:
            continue
        raw = indexed[job['upstream_id']]
        if any(raw.get(k) != v for k, v in job.items() if k not in ('id', 'upstream_id', 'vector_id', 'format')):
            raise ValueError('复用回答与固定身份或提示不符')
        result.append({**job, **{k: raw[k] for k in ('prompt', 'a', 'token_ids', 'n_generated_tokens', 'finish_reason')}})
    if len(result) != 90:
        raise ValueError('复用回答必须恰为90条')
    return result


def score_order(rows, pole):
    ids = [r['id'] for r in rows if r['pole'] == pole]
    random.Random(0).shuffle(ids)
    return ids


def valid(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100


def summarize(rows, jobs, capability):
    """每题先平均五中性，再计算配对误差；缺分/能力失败保留。"""
    if len(rows) != len(jobs) or any(any(row.get(k) != v for k, v in job.items()) for row, job in zip(rows, jobs)):
        raise ValueError('行为回答清单不完整或身份、顺序改变')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('行为回答身份重复')
    bootstrap = np.random.default_rng(0).integers(0, 5, size=(10000, 5))
    def stats(values):
        return {'mean': float(values.mean()), 'per_question': values.tolist(),
                'question_bootstrap_ci': np.quantile(values[bootstrap].mean(1), [.025, .975]).tolist()}
    reports = []
    for pole in ('high', 'low'):
        def array(arm, metric, a=None, r=None, fmt=None):
            selected = [row for row in rows if row['pole'] == pole and row['arm'] == arm and
                        (arm == 'neutral' or (row['action_sign'], row['outcome_sign'], row['format']) == (a, r, fmt))]
            count = 25 if arm in ('neutral', 'injected') else 5
            if len(selected) != count or any(not valid(row.get(metric)) for row in selected):
                return None
            values = []
            for q in range(5):
                group = [row for row in selected if row['question_index'] == q]
                if len(group) != (5 if count == 25 else 1):
                    raise ValueError('题目或中性覆盖不完整')
                if count == 25 and {row['memory_index'] for row in group} != set(range(5)):
                    raise ValueError('五中性身份重复或遗漏')
                values.append(np.mean([row[metric] for row in group]))
            return np.array(values)
        record = {'pole': pole, 'metrics': {}}
        for metric in METRICS:
            neutral = array('neutral', metric)
            contexts = {(a, r): array('context', metric, a, r) for a, r in CELLS}
            orientation = -1 if pole == 'low' and metric == 'trait' else 1
            def contrast(cells):
                if any(v is None for v in cells.values()):
                    return {'status': 'insufficient'}
                dp, dm = cells[1, 1] - cells[1, -1], cells[-1, 1] - cells[-1, -1]
                return {'status': 'complete', 'd_plus': stats(dp), 'd_minus': stats(dm),
                        'd_plus_high_axis': stats(orientation * dp), 'd_minus_high_axis': stats(orientation * dm),
                        'interaction_raw': stats((dp - dm) / 2), 'valence_raw': stats((dp + dm) / 2),
                        'interaction_high_axis': stats(orientation * (dp - dm) / 2),
                        'valence_high_axis': stats(orientation * (dp + dm) / 2)}
            result = {'neutral': stats(neutral) if neutral is not None else None,
                      'context_cells': {f'a{a}_r{r}': {
                          'absolute': stats(value) if value is not None else None,
                          'minus_neutral': stats(value - neutral) if value is not None and neutral is not None else None}
                          for (a, r), value in contexts.items()},
                      'context_contrast': contrast(contexts), 'formats': {}}
            for fmt in FORMATS:
                injected = {(a, r): array('injected', metric, a, r, fmt) for a, r in CELLS}
                cells = {}
                for a, r in CELLS:
                    vid = f'a{a}_r{r}_{fmt}'
                    value, target, mismatch = injected[a, r], contexts[a, r], injected[a, -r]
                    cell = {'capability': capability[vid], 'absolute': stats(value) if value is not None else None,
                            'status': 'capability_failure' if not capability[vid]['within_2pp'] else
                                      'incomplete_scores' if value is None or neutral is None else 'measured',
                            'minus_neutral': stats(value - neutral) if value is not None and neutral is not None else None,
                            'match_comparison': {'status': 'insufficient'}}
                    if metric in ('coherence', 'task') and value is not None and neutral is not None:
                        cell['quality'] = {'loss': stats(neutral - value),
                                           'within_10_loss': float((neutral - value).mean()) <= 10,
                                           'absolute_requirement': '>=75' if metric == 'coherence' else 'not_specified',
                                           'absolute_satisfied': bool(value.mean() >= 75) if metric == 'coherence' else None}
                    if all(v is not None for v in (value, target, mismatch, neutral)):
                        target_effect, match_effect, mismatch_effect = target - neutral, value - neutral, mismatch - neutral
                        error, wrong_error = np.abs(match_effect - target_effect), np.abs(mismatch_effect - target_effect)
                        cell['match_comparison'] = {'status': 'complete', 'target_effect': stats(orientation * target_effect),
                            'matched_effect': stats(orientation * match_effect), 'mismatched_effect': stats(orientation * mismatch_effect),
                            'matched_absolute_error': stats(error), 'mismatched_absolute_error': stats(wrong_error),
                            'mismatch_minus_match_error': stats(wrong_error - error),
                            'matched_closer_on_mean': bool(error.mean() < wrong_error.mean())}
                    cells[f'a{a}_r{r}'] = cell
                result['formats'][fmt] = {'cells': cells, 'contrast': contrast(injected)}
            record['metrics'][metric] = result
        reports.append(record)
    return {'scope': 'raw_twin_single_state_dev_five_question_one_seed', 'reports': reports,
            'n_answers': len(rows), 'missing_score_assignments': sum(not valid(r.get(m)) for r in rows for m in METRICS),
            'finish_counts': {arm: {reason: sum(r['arm'] == arm and r['finish_reason'] == reason for r in rows)
                                   for reason in ('eos', 'length')} for arm in ('neutral', 'context', 'injected')},
            'capability': capability, 'formal_e1_pass': False, 'behavior_validated': False,
            'limits': '一个已选开发情境、五题、一种子；区间只覆盖题目，不计算保留率，不授予正式行为资格。'}
