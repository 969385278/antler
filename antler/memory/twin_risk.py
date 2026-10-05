"""C12：四单元的正交分解及正确实验单位下的有限置换。"""
from itertools import product

import torch

from .twin_state import cosine


def decompose(states):
    """顺序为--、-+、+-、++；系数使用标准二水平因子编码。"""
    if states.ndim != 2 or states.shape[0] != 4 or not torch.isfinite(states).all():
        raise ValueError('需要四个有限状态向量')
    h = states.double()
    return {'mean': h.mean(0), 'action': (-h[0]-h[1]+h[2]+h[3])/4,
            'feedback': (-h[0]+h[1]-h[2]+h[3])/4,
            'interaction': (h[0]-h[1]-h[2]+h[3])/4}


def geometry(activations, scale):
    """整篇故事统一换标签，绝不逐探针随机交换以伪造独立样本。"""
    if activations.shape[:2] != (9, 60) or not scale > 0:
        raise ValueError('激活预算或尺度不符')
    h = (activations[:4].double()-activations[4:].double().mean(0))/scale
    all_states = h.mean(1)
    terms = decompose(all_states)
    null = []
    for left, right in product((False, True), repeat=2):
        order = [1, 0] if left else [0, 1]
        order += [3, 2] if right else [2, 3]
        permuted = all_states[order]
        term = decompose(permuted)['interaction']
        null.append({'swap_negative_action': left, 'swap_positive_action': right,
                     'interaction_norm_s': float(term.norm()),
                     'feedback_difference_cosine': cosine(permuted[3]-permuted[2], permuted[1]-permuted[0])})
    first, second = decompose(h[:, :30].mean(1)), decompose(h[:, 30:].mean(1))
    return {'terms': {k: {'vector_s': v.tolist(), 'norm_s': float(v.norm())} for k, v in terms.items()},
            'probe_half_cosines': {k: cosine(first[k], second[k]) for k in terms},
            'exact_story_relabelings': null, 'unique_relabelings': 4,
            'formal_p_value': None,
            'limits': '只有四篇固定故事，无独立改写；四种标签交换仅作敏感性诊断，不能重复1000次当作新的证据。标准交互系数为旧报告I的一半。'}
