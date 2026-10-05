"""C12：原始单段状态、结果差值和幅度的固定诊断。"""
from __future__ import annotations

import random

import torch

from .write import projection_coordinates


CELLS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def select_probes(pool):
    """保存来源身份，固定打乱前60题；两个30题组不重复题目。"""
    if len(pool) < 60 or any(not isinstance(p, dict) or not isinstance(p.get('question'), str)
                             or not p['question'].strip() for p in pool):
        raise ValueError('提取探针池不足或文本无效')
    if len({p['question'] for p in pool}) != len(pool):
        raise ValueError('提取探针池含重复文本，不能称为不重叠探针组')
    result = list(pool)
    random.Random(0).shuffle(result)
    return result[:60]


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return float(a @ b / denominator) if float(denominator) > 0 else None


def describe(vector, basis, scale):
    """非正交基的重建范数与系数范数分开；保留全部五轴。"""
    delta = projection_coordinates(vector, basis, 1., projection='dual').double()
    u = basis.double()
    rebuilt = delta @ u
    residual = vector.double() - rebuilt
    return {'full_state': vector.tolist(), 'dual_coordinates': delta.tolist(),
            'axis_projection': (u @ vector.double()).tolist(),
            'reconstructed_state': rebuilt.tolist(), 'full_norm_s': float(vector.norm()),
            'reconstructed_norm_s': float(rebuilt.norm()), 'residual_norm_s': float(residual.norm()),
            'full_norm_physical': float(vector.norm()) * scale,
            'reconstructed_norm_physical': float(rebuilt.norm()) * scale}


def analyze(activations, basis, scale, *, layer=16):
    """输入九条件×60探针×隐藏维；前四为CELLS、后五为固定中性。"""
    if (not isinstance(activations, torch.Tensor) or activations.ndim != 3
            or activations.shape[:2] != (9, 60) or not activations.is_floating_point()
            or not torch.isfinite(activations).all()):
        raise ValueError('必须保存九条件×60探针的完整有限激活')
    # 复用对偶解算器检查基、维度、尺度和满秩。
    projection_coordinates(activations[0, 0], basis, scale)
    acts = activations.detach().double().cpu()
    baseline = acts[4:].mean(0)
    per_probe = (acts[:4] - baseline) / scale
    groups, vectors = {}, {}
    for name, selection in (('first30', slice(0, 30)), ('second30', slice(30, 60)),
                            ('all60', slice(0, 60))):
        h = per_probe[:, selection].mean(1)
        values = {f'a{a}_r{r}': h[i] for i, (a, r) in enumerate(CELLS)}
        values['d_plus'] = h[3] - h[2]
        values['d_minus'] = h[1] - h[0]
        values['interaction'] = (values['d_plus'] - values['d_minus']) / 2
        values['valence'] = (values['d_plus'] + values['d_minus']) / 2
        vectors[name] = values
        groups[name] = {key: describe(value, basis.cpu(), scale) for key, value in values.items()}
    stability = {}
    for key in vectors['all60']:
        first, second = vectors['first30'][key], vectors['second30'][key]
        delta1 = torch.tensor(groups['first30'][key]['reconstructed_state'], dtype=torch.float64)
        delta2 = torch.tensor(groups['second30'][key]['reconstructed_state'], dtype=torch.float64)
        stability[key] = {'full_cosine': cosine(first, second),
                          'reconstructed_cosine': cosine(delta1, delta2),
                          'second_over_first_norm': float(second.norm() / first.norm())
                          if float(first.norm()) > 0 else None}
    return {'layer': layer, 'dims': ['O', 'C', 'E', 'A', 'N'], 'scale': scale,
            'groups': groups, 'probe_stability': stability, 'formal_e1_pass': False,
            'behavior_validated': False, 'injection_performed': False,
            'limits': '结果差值是配对诊断；两组探针不是独立经历。未优化、放大或测试行为。'}
