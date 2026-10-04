"""C12：结果双胞胎的固定设计与行动×反馈交互汇总。"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from antler.memory.write import BASE_SYSTEM, MEMORY_TEMPLATE, NEUTRAL_MEMORIES
from antler.traits.data import BIG_FIVE, load_trait


DIMS_TESTED = ('O', 'E', 'A')
SCENARIO_IDS = ('twin01_study', 'twin02_walking', 'twin03_language',
                'twin04_ideas', 'twin05_garden', 'twin06_clubroom')
SEEDS = (0, 1, 2)
N_ANSWERS = 2520
MAX_TOKENS = 768


def expand(data):
    """同一行动仅替换结局；两种行动共享结局，不添加未来行为结论。"""
    if (not isinstance(data, dict) or data.get('scope') != 'twin_outcome_dev_v1' or data.get('split') != 'dev'
            or data.get('provenance') != 'assistant_authored' or data.get('formal_d1_data') is not False):
        raise ValueError('材料必须是独立助手起草的固定开发诊断')
    scenes = data.get('scenarios')
    if (not isinstance(scenes, list) or len(scenes) != 6 or any(not isinstance(s, dict) for s in scenes)
            or tuple(s.get('id') for s in scenes) != SCENARIO_IDS
            or tuple(s.get('dim') for s in scenes) != ('O', 'O', 'E', 'E', 'A', 'A')):
        raise ValueError('情境数量、身份、次序或目标改变')
    episodes = []
    for scene in scenes:
        prefix = scene.get('prefix')
        if not isinstance(prefix, str) or not prefix.strip() or prefix != prefix.strip():
            raise ValueError('共享情境必须是非空且无首尾空白的文本')
        for name in ('actions', 'outcomes'):
            parts = scene.get(name)
            if (not isinstance(parts, dict) or set(parts) != {'-1', '1'}
                    or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in parts.values())
                    or parts['1'] == parts['-1']):
                raise ValueError('行动和结局须各包含两个不同的固定版本')
        for action in (-1, 1):
            for outcome in (-1, 1):
                text = ' '.join((prefix, scene['actions'][str(action)], scene['outcomes'][str(outcome)]))
                if any(term in text.lower() for term in (
                    'from then on', 'from now on', 'after that i', 'i began', 'i learned', 'in the future')):
                    raise ValueError('故事不能直接给出未来行为结论')
                episodes.append({'id': f'{scene["id"]}_a{action}_r{outcome}', 'scenario_id': scene['id'],
                                 'dim': scene['dim'], 'action_sign': action, 'outcome_sign': outcome,
                                 'predicted_sign': action * outcome, 'text': text,
                                 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    if len({e['text_sha256'] for e in episodes}) != 24:
        raise ValueError('二十四个实验单元不能重复')
    return episodes


def jobs_for(data):
    episodes = expand(data)
    jobs = []
    for dim in DIMS_TESTED:
        for pole in ('high', 'low'):
            trait = load_trait(BIG_FIVE[dim][pole == 'low'], 'eval')
            if len(trait['questions']) != 20:
                raise ValueError('评测题数不符固定切分')
            arms = [('none', None, None, BASE_SYSTEM)]
            arms += [('neutral', i, None, MEMORY_TEMPLATE.format(experience=m))
                     for i, m in enumerate(NEUTRAL_MEMORIES)]
            arms += [('context', None, e, MEMORY_TEMPLATE.format(experience=e['text']))
                     for e in episodes if e['dim'] == dim]
            for arm, memory, episode, system in arms:
                for seed in SEEDS:
                    for q in range(10):
                        jobs.append({'id': len(jobs), 'dim': dim, 'pole': pole, 'split': 'calib',
                                     'arm': arm, 'memory_index': memory,
                                     'episode_id': episode['id'] if episode else None,
                                     'scenario_id': episode['scenario_id'] if episode else None,
                                     'action_sign': episode['action_sign'] if episode else None,
                                     'outcome_sign': episode['outcome_sign'] if episode else None,
                                     'seed': seed, 'sample_index': seed, 'question_index': q,
                                     'q': trait['questions'][q], 'system': system})
    if len(jobs) != N_ANSWERS:
        raise ValueError('固定回答预算不一致')
    return jobs


def _valid(value):
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 100


def summarize(rows, data):
    expected = jobs_for(data)
    if len(rows) != len(expected) or any(any(row.get(k) != v for k, v in job.items())
                                       for row, job in zip(rows, expected)):
        raise ValueError('完整回答身份、次序或提示与固定设计不一致')
    indexed = {(r['dim'], r['pole'], r['arm'], r['memory_index'], r['episode_id'],
                r['question_index'], r['seed']): r for r in rows}
    indices = np.random.default_rng(0).integers(0, 10, size=(10000, 10))
    episodes = expand(data)
    reports = []
    for scene in data['scenarios']:
        poles = []
        for pole in ('high', 'low'):
            dim = scene['dim']
            def group(arm, memory=None, episode=None):
                return [indexed[dim, pole, arm, memory, episode, q, s] for q in range(10) for s in SEEDS]
            cells = {(a, r): group('context', episode=f'{scene["id"]}_a{a}_r{r}')
                     for a in (-1, 1) for r in (-1, 1)}
            neutral, none = [group('neutral', i) for i in range(5)], group('none')
            selected = [r for cell in cells.values() for r in cell] + none + [r for g in neutral for r in g]
            missing = sum(any(not _valid(row.get(m)) for m in ('trait', 'coherence', 'task')) for row in selected)
            record = {'scenario_id': scene['id'], 'dim': dim, 'pole': pole,
                      'missing_or_invalid_rows': missing, 'metrics': {}, 'gate': 'incomplete',
                      'finish_counts': {f'a{a}_r{r}': {
                          reason: sum(row.get('finish_reason') == reason for row in cell)
                          for reason in ('eos', 'length')} for (a, r), cell in cells.items()}}
            def stats(matrix):
                qmeans = matrix.mean(1)
                return {'mean': float(qmeans.mean()),
                        'question_bootstrap_ci': np.quantile(qmeans[indices].mean(1), [.025, .975]).tolist(),
                        'per_seed_means': matrix.mean(0).tolist()}
            for metric in ('trait', 'coherence', 'task'):
                bad = sum(not _valid(row.get(metric)) for row in selected)
                if bad:
                    record['metrics'][metric] = {'status': 'incomplete', 'missing_or_invalid_rows': bad}
                    continue
                matrix = lambda group_: np.array([row[metric] for row in group_], dtype=np.float64).reshape(10, 3)
                c = {key: matrix(cell) for key, cell in cells.items()}
                baseline = np.mean([matrix(g) for g in neutral], axis=0)
                no_memory = matrix(none)
                d_plus, d_minus = c[1, 1] - c[1, -1], c[-1, 1] - c[-1, -1]
                interaction, valence = (d_plus - d_minus) / 2, (d_plus + d_minus) / 2
                report = {'status': 'complete', 'neutral': stats(baseline), 'none': stats(no_memory),
                          'cells': {f'a{a}_r{r}': {'absolute': stats(value),
                                    'minus_neutral': stats(value - baseline), 'minus_none': stats(value - no_memory)}
                                    for (a, r), value in c.items()},
                          'feedback_positive_action': stats(d_plus), 'feedback_negative_action': stats(d_minus),
                          'interaction_raw': stats(interaction), 'valence_raw': stats(valence)}
                if metric == 'trait':
                    orientation = 1 if pole == 'high' else -1
                    report['interaction_high_axis'] = stats(orientation * interaction)
                    report['valence_high_axis'] = stats(orientation * valence)
                    low, high = report['interaction_high_axis']['question_bootstrap_ci']
                    record['trait_gate'] = 'expected' if low > 0 else 'opposite' if high < 0 else 'not_established'
                record['metrics'][metric] = report
            if not missing:
                record['gate'] = record['trait_gate']
            poles.append(record)
        gates = [p['gate'] for p in poles]
        category = ('incomplete' if 'incomplete' in gates else 'opposite' if 'opposite' in gates
                    else 'established' if gates == ['expected', 'expected']
                    else 'partial' if 'expected' in gates else 'not_established')
        reports.append({'scenario_id': scene['id'], 'dim': scene['dim'], 'gate': category, 'poles': poles})
    return {'scope': 'six_scenario_action_outcome_context_gate_dev_calib', 'reports': reports,
            'n_scenarios': 6, 'n_stories': len(episodes), 'n_answers': len(rows),
            'gate_counts': {name: sum(r['gate'] == name for r in reports)
                            for name in ('established', 'partial', 'opposite', 'not_established', 'incomplete')},
            'formal_e1_pass': False, 'behavior_validated': False, 'injection_performed': False,
            'limits': ['仅测O/E/A；题目区间不覆盖情境泛化，无多重比较正式结论。',
                       '三种子在固定批次内配对，不把回答计作独立经历。',
                       '上下文交互不等于压缩、人格读数、能力或注入资格通过。',
                       '全部四单元和两极完整报告，截断与质量代价不能由人格增益豁免。']}
