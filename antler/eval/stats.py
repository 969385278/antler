"""C8：按题目聚合的统计工具，不把同题重复生成当作独立样本。"""
import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np


def question_means(values, question_ids):
    """显式排除缺失值；保留有有效分数的题目，不以零分填补。"""
    if len(values) != len(question_ids):
        raise ValueError('分数与题号长度不一致')
    groups = {}
    for value, question in zip(values, question_ids):
        if value is None:
            continue
        if not math.isfinite(value):
            raise ValueError('分数必须有限，缺失值请使用None')
        groups.setdefault(question, []).append(float(value))
    if not groups:
        raise ValueError('没有有效题目')
    return {q: float(np.mean(v)) for q, v in groups.items()}


def bootstrap_mean(values, question_ids, *, n_boot=10000, confidence=.95, seed=0):
    means = question_means(values, question_ids)
    return _bootstrap(list(means.values()), n_boot, confidence, seed)


def _bootstrap(values, n_boot, confidence, seed):
    if not 0 < confidence < 1 or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError('置信度或重采样次数无效')
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        raise ValueError('至少需要两个独立题目才能估计区间')
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(n_boot, len(data)))].mean(axis=1)
    low, high = np.quantile(means, [(1-confidence)/2, (1+confidence)/2])
    return {'mean': float(data.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'n_questions': len(data), 'n_boot': n_boot, 'confidence': confidence, 'seed': seed}


def paired_bootstrap(a, a_ids, b, b_ids, *, n_boot=10000, confidence=.95, seed=0):
    """报告b−a；仅配对两侧均有效的题目，同时报告被排除题数。"""
    left, right = question_means(a, a_ids), question_means(b, b_ids)
    shared = [q for q in left if q in right]
    differences = [right[q] - left[q] for q in shared]
    result = _bootstrap(differences, n_boot, confidence, seed)
    # 点估计用精确十进制分数计算，避免恰好下降10分被浮点误判为超限。
    # 仍然每题等权；不改变重采样分布、区间或缺失题处理。
    def exact_means(values, ids):
        groups = {}
        for value, question in zip(values, ids):
            if value is not None and question in shared:
                groups.setdefault(question, []).append(Fraction(str(float(value))))
        return {q: sum(v, Fraction(0)) / len(v) for q, v in groups.items()}

    exact_left, exact_right = exact_means(a, a_ids), exact_means(b, b_ids)
    result['mean'] = float(sum((exact_right[q] - exact_left[q] for q in shared),
                               Fraction(0)) / len(shared))
    sd = float(np.std(differences, ddof=1))
    result.update(n_unpaired_questions=len(set(left) | set(right)) - len(shared),
                  effect_size_dz=result['mean'] / sd if sd > 0 else None)
    return result


def wilson_interval(successes, total, *, confidence=.95):
    if (not isinstance(total, int) or not isinstance(successes, int) or total <= 0
            or not 0 <= successes <= total or not 0 < confidence < 1):
        raise ValueError('成功数、总数或置信度无效')
    z = NormalDist().inv_cdf((1+confidence)/2)
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return {'proportion': p, 'ci_low': max(0., center-half), 'ci_high': min(1., center+half), 'n': total}


def benjamini_hochberg(p_values, *, q=.05):
    """返回原顺序的校正p值和拒绝标志；缺失值不进入检验族。"""
    if not 0 < q < 1:
        raise ValueError('q必须在0和1之间')
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for _, p in valid):
        raise ValueError('p值必须在0和1之间')
    ordered = sorted(valid, key=lambda t: t[1])
    adjusted = [None] * len(p_values)
    bound = 1.
    for rank in range(len(ordered), 0, -1):
        index, p = ordered[rank-1]
        bound = min(bound, p * len(ordered)/rank)
        adjusted[index] = bound
    return {'adjusted_p': adjusted, 'reject': [p is not None and p <= q for p in adjusted], 'q': q}
