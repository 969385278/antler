"""C4：校准曲线及逐方向数字门槛；不据此宣布正式E0通过。"""
from collections import Counter
import math

import numpy as np

from .stats import paired_bootstrap, question_means


def spearman(x, y):
    """平均秩处理并列值；常量或不足两个点时不可定义。"""
    if len(x) != len(y) or len(x) < 2 or not all(math.isfinite(v) for v in [*x, *y]):
        return None
    def rank(values):
        array = np.asarray(values)
        return np.array([sum(array < v) + (sum(array == v)+1)/2 for v in array])
    a, b = rank(x), rank(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def assess_point(baseline, injected):
    """输入对应目标极点的样本；区间按题目重采样，缺分不通过数字门槛。"""
    if not baseline or not injected:
        raise ValueError('基线和注入样本均不能为空')
    left_ids = [s['q'] for s in baseline]
    right_ids = [s['q'] for s in injected]
    counts_a, counts_b = Counter(left_ids), Counter(right_ids)
    complete = (counts_a == counts_b and len(counts_a) == 10 and all(n >= 3 for n in counts_a.values()))
    metrics = {}
    for metric in ['trait','coherence','task']:
        a, b = [s.get(metric) for s in baseline], [s.get(metric) for s in injected]
        if any(v is None for v in [*a,*b]):
            complete = False
        try:
            estimate = paired_bootstrap(a,left_ids,b,right_ids)
            estimate['baseline'] = float(np.mean(list(question_means(a,left_ids).values())))
            estimate['injected'] = float(np.mean(list(question_means(b,right_ids).values())))
            metrics[metric] = estimate
        except ValueError:
            metrics[metric] = None
            complete = False
    t,c,q = [metrics[k] for k in ['trait','coherence','task']]
    checks = {'complete_samples': complete,
              'trait_gain_ge20': t is not None and t['mean'] >= 20,
              'trait_ci_low_gt10': t is not None and t['ci_low'] > 10,
              'coherence_drop_le10': c is not None and c['mean'] >= -10,
              'coherence_absolute_ge75': c is not None and c['injected'] >= 75,
              'task_drop_le10': q is not None and q['mean'] >= -10}
    return {'metrics':metrics, 'checks':checks, 'numeric_criteria_met':all(checks.values()),
            'failed_checks':[name for name, passed in checks.items() if not passed],
            'formal_e0_pass':False,
            'note':'数字门槛不替代test、独立种子、MMLU、串扰矩阵和人工校验'}
