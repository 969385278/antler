"""C11：同方法、同坐标系、不同探针的δ重复测量描述统计。"""
import math
from statistics import mean

from ..traits.data import DIMS


def compare_probe_replicates(reference, repeat):
    """不把A/B一致性或重复同一输出当作探针稳定性，不设置事后通过线。"""
    for key in ('method', 'layer', 'basis_sha256', 'units', 'fit_settings'):
        if key not in reference or reference[key] != repeat.get(key):
            raise ValueError('重复测量的方法、层、坐标基和单位必须相同')
    if reference['method'] not in ('A', 'B') or reference['units'] != 's_layer':
        raise ValueError('未知方法或坐标单位')
    normalize = lambda seq: {' '.join(s.casefold().split()) for s in seq}
    a_probes = reference['probes']; b_probes = repeat['probes']
    if (not a_probes or len(a_probes) != len(b_probes)
            or len(normalize(a_probes)) != len(a_probes)
            or len(normalize(b_probes)) != len(b_probes)
            or normalize(a_probes) & normalize(b_probes)):
        raise ValueError('重复探针须等量、各自去重且互不重叠')
    a_rows, b_rows = reference['episodes'], repeat['episodes']
    a = {r['id']: r for r in a_rows}; b = {r['id']: r for r in b_rows}
    if not a or len(a) != len(a_rows) or len(b) != len(b_rows) or set(a) != set(b):
        raise ValueError('必须逐经历完整配对，不允许缺失或重复')

    def vector(row):
        values = row['delta']
        if len(values) != 5 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                   or not math.isfinite(v) for v in values):
            raise ValueError('δ必须按O/C/E/A/N提供五个有限数值')
        return values

    sign = lambda v: (v > 0) - (v < 0)
    records = []
    by_dim = {d: {'target_count': 0, 'sign_agree': 0, 'both_correct': 0} for d in DIMS}
    for identity in a:
        first, second = a[identity], b[identity]
        for key in ('text_sha256', 'group_id', 'intended'):
            if key not in first or first[key] != second.get(key):
                raise ValueError('经历原文、家族或预期目标改变')
        labels = first['intended']
        if any(d not in DIMS or type(s) is not int or s not in (-1, 1) for d, s in labels.items()):
            raise ValueError('预期方向非法')
        x, y = vector(first), vector(second)
        nx, ny = math.sqrt(sum(v*v for v in x)), math.sqrt(sum(v*v for v in y))
        cosine = max(-1., min(1., sum(v*w for v, w in zip(x, y))/(nx*ny))) if nx and ny else None
        for d, expected in labels.items():
            j = DIMS.index(d)
            by_dim[d]['target_count'] += 1
            by_dim[d]['sign_agree'] += int(sign(x[j]) == sign(y[j]))
            by_dim[d]['both_correct'] += int(sign(x[j]) == sign(y[j]) == expected)
        records.append({'id': identity, 'group_id': first['group_id'], 'cosine': cosine,
                        'reference_norm': nx, 'repeat_norm': ny,
                        'norm_ratio': ny/nx if nx else None,
                        'relative_l2': math.sqrt(sum((v-w)**2 for v, w in zip(x, y)))/nx if nx else None,
                        'coordinate_sign_agreement': sum(sign(v) == sign(w) for v, w in zip(x, y))/5})
    valid = [r['cosine'] for r in records if r['cosine'] is not None]
    return {'scope': 'probe_replicate_description', 'method': reference['method'],
            'n_episodes': len(records), 'n_groups': len({r['group_id'] for r in records}),
            'cosine_mean': mean(valid) if valid else None,
            'cosine_undefined': len(records)-len(valid), 'per_episode': records,
            'target_signs': by_dim, 'formal_e1_pass': False,
            'limitation': '同家族经历不独立；零向量余弦未定义；符号一致不等于方向正确；不以此替代行为验证'}
