"""C11：完整12段pilot的分布保留与语义方向汇总，不作正式通过判定。"""
import math

import numpy as np

from ..traits.data import DIMS


def _distribution(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return {'mean': None, 'episode_bootstrap_ci': None, 'undefined': len(values)}
    a = np.asarray(valid, dtype=np.float64)
    rng = np.random.default_rng(0)
    boot = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(1)
    return {'mean': float(a.mean()), 'episode_bootstrap_ci': np.quantile(boot, [.025, .975]).tolist(),
            'undefined': len(values) - len(valid), 'n_episodes': len(valid)}


def summarize(rows):
    """按经历而不是按16个标签重采样；保留负解释率与所有失败案例。"""
    if len(rows) != 12 or [r.get('id') for r in rows] != [f'p{i:02}' for i in range(1, 13)]:
        raise ValueError('必须是完整、固定顺序的12段pilot')
    if sum(len(r['intended']) for r in rows) != 16:
        raise ValueError('固定16个预期标签缺失')
    report = {'formal_e1_pass': False, 'scope': 'unbalanced_pilot_not_formal_D1', 'episodes': [],
              'arms': {}, 'semantic': {}, 'n_episodes': 12}
    for row in rows:
        if set(row['arms']) != {'personality', 'random', 'full'}:
            raise ValueError('三个拟合臂必须齐全')
        ref = row['arms']['personality']['baseline']
        if any(v['baseline'] != ref for v in row['arms'].values()):
            raise ValueError('各臂没有共享同一完整基线')
        baseline_ids = [p['probe'] for p in ref['per_probe']]
        if len(baseline_ids) != 10 or len(set(baseline_ids)) != 10:
            raise ValueError('必须保留10道唯一留出探针')
        conditions = {'A': row['A_eval'], **{k: v['final'] for k, v in row['arms'].items()}}
        metrics = {}
        for name, condition in conditions.items():
            if [p['probe'] for p in condition['per_probe']] != baseline_ids:
                raise ValueError('最终评估探针与基线不一致')
            for value in (condition, ref):
                if not math.isfinite(value['kl']) or any(not math.isfinite(p['kl']) for p in value['per_probe']):
                    raise ValueError('KL非有限')
                if not math.isclose(value['kl'], sum(p['kl'] for p in value['per_probe']) / 10, abs_tol=1e-7):
                    raise ValueError('整体KL与逐题平均不符')
            metrics[name] = {'kl': condition['kl'], 'kl_reduction': ref['kl'] - condition['kl'],
                             'explained': (ref['kl'] - condition['kl']) / ref['kl'] if ref['kl'] > 0 else None,
                             'improved_probes': sum(a['kl'] < b['kl'] for a, b in zip(condition['per_probe'], ref['per_probe']))}
            if name != 'A':
                vector = row['arms'][name]['vector']
                if not vector or not all(math.isfinite(x) for x in vector):
                    raise ValueError('实际注入向量非法')
                metrics[name]['vector_norm'] = math.sqrt(sum(x*x for x in vector))
        report['episodes'].append({'id': row['id'], 'baseline_kl': ref['kl'], 'conditions': metrics})
    for name in ('A', 'personality', 'random', 'full'):
        report['arms'][name] = {metric: _distribution([r['conditions'][name][metric] for r in report['episodes']])
                                for metric in ('kl_reduction', 'explained')}
    report['personality_minus_random_explained'] = _distribution([
        e['conditions']['personality']['explained'] - e['conditions']['random']['explained']
        if e['conditions']['personality']['explained'] is not None else None for e in report['episodes']])
    for method in ('A', 'B'):
        counts = {d: {'hits': 0, 'total': 0, 'positive_labels': 0, 'negative_labels': 0} for d in DIMS}
        per_episode = []
        for row in rows:
            coords = row['delta_A'] if method == 'A' else row['arms']['personality']['coordinates']
            if len(coords) != 5 or not all(math.isfinite(x) for x in coords):
                raise ValueError('人格坐标须为有限五维')
            for dim, sign in row['intended'].items():
                if dim not in DIMS or type(sign) is not int or sign not in (-1, 1):
                    raise ValueError('预期标签非法')
                counts[dim]['total'] += 1
                counts[dim]['hits'] += int(coords[DIMS.index(dim)] * sign > 0)
                counts[dim]['positive_labels' if sign > 0 else 'negative_labels'] += 1
            per_episode.append({'id': row['id'], 'coordinates': coords, 'intended': row['intended'],
                                'intended_absolute_ranks': {dim: 1 + sum(abs(v) > abs(coords[DIMS.index(dim)]) for v in coords)
                                                            for dim in row['intended']}})
        report['semantic'][method] = {'per_dimension': counts, 'hits': sum(v['hits'] for v in counts.values()),
                                     'total_labels': 16, 'independent_episode_count': 12, 'per_episode': per_episode,
                                     'rank_ties': '相同绝对值并列；排序与符号正确性分开报告'}
    return report
