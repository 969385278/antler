"""C11：五中性记忆配对行为汇总，不执行生成或授予行为验证资格。"""
import math
from numbers import Real

import numpy as np

from ..traits.data import DIMS


METRICS = ('trait', 'coherence', 'task')


def summarize_behavior(baselines, context, injected, *, intended_sign, seed=0):
    """汇总一个经历、一个维度/极点、一个冻结注入条件的test评分。

    baselines与injected分别按固定五条NEUTRAL_MEMORIES顺序传入
    trait_score完整输出。调用方负责绑定提示、模型、注入条件和原始回答；
    此函数只核验评分配对，不能证明采集来源或替代条件验收。
    缺分直接拒绝，禁止各臂分别删样本后比较。
    """
    if type(intended_sign) is not int or intended_sign not in (-1, 1):
        raise ValueError('预定方向必须为+1或-1')
    if len(baselines) != 5 or len(injected) != 5:
        raise ValueError('基线和注入均须包含按固定顺序排列的五条中性记忆')
    if context.get('dim') not in DIMS or context.get('pole') not in ('high', 'low'):
        raise ValueError('维度或极点非法')
    identity = (context['dim'], context['pole'], 'test')

    def index(report):
        if (report.get('dim'), report.get('pole'), report.get('split')) != identity:
            raise ValueError('必须使用同一维度、极点和test子集')
        rows = {}
        for row in report['samples']:
            q, sample = row['question_index'], row['sample_index']
            if type(q) is not int or type(sample) is not int or q not in range(10, 20) or sample < 0:
                raise ValueError('test题号或采样编号非法')
            key = (q, sample)
            if key in rows or not isinstance(row.get('q'), str) or not row['q'].strip():
                raise ValueError('评分身份重复或题目为空')
            for metric in METRICS:
                value = row.get(metric)
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError('人格、连贯性及任务分必须完整且在0到100之间')
            rows[key] = row
        n = max((key[1] for key in rows), default=-1) + 1
        if not n or set(rows) != {(q, s) for q in range(10, 20) for s in range(n)}:
            raise ValueError('须保留全部十道test题及每题相同的完整采样编号')
        return rows

    teacher = index(context)
    base = [index(r) for r in baselines]
    steer = [index(r) for r in injected]
    for rows in base + steer:
        if set(rows) != set(teacher) or any(rows[k]['q'] != teacher[k]['q'] for k in teacher):
            raise ValueError('各臂题目、题号与采样身份须完整配对')

    # 先在每题内平均采样，再平均五条中性记忆，最后平均题目。
    n_samples = len(teacher) // 10
    def values(rows):
        return np.array([[np.mean([rows[q, s][m] for s in range(n_samples)])
                          for m in METRICS] for q in range(10, 20)], dtype=np.float64)

    neutral = np.mean([values(r) for r in base], axis=0)
    written = np.mean([values(r) for r in steer], axis=0)
    contextual = values(teacher)
    context_delta = contextual - neutral
    injection_delta = written - neutral
    denominator = float(context_delta[:, 0].mean())
    numerator = float(injection_delta[:, 0].mean())
    ratio = numerator / denominator if denominator != 0 else None

    # 三臂、五记忆与重复采样按题成组重采样，不能当成独立题增加样本量。
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 10, size=(10000, 10))
    boot_context = context_delta[indices].mean(axis=1)
    boot_injection = injection_delta[indices].mean(axis=1)
    interval = lambda a: np.quantile(a, [.025, .975], axis=0).tolist()
    denominator_ci = interval(boot_context[:, 0])
    ratio_ci = None
    ratio_reason = None
    if denominator == 0:
        ratio_reason = '上下文人格效果为零，迁移比未定义'
    elif denominator_ci[0] <= 0 <= denominator_ci[1] or np.any(boot_context[:, 0] == 0):
        ratio_reason = '上下文效果区间包含零或重采样分母为零，不报告有限比值区间'
    else:
        ratio_ci = interval(boot_injection[:, 0] / boot_context[:, 0])
    expected = intended_sign if context['pole'] == 'high' else -intended_sign
    means = lambda a: {m: float(a[:, j].mean()) for j, m in enumerate(METRICS)}
    return {
        'scope': 'one_episode_one_condition_paired_test_summary',
        'dim': context['dim'], 'pole': context['pole'], 'intended_sign': intended_sign,
        'n_questions': 10, 'n_samples_per_question': n_samples, 'n_neutral_memories': 5,
        'neutral': means(neutral), 'context': means(contextual), 'injected': means(written),
        'context_effect': means(context_delta), 'injected_effect': means(injection_delta),
        'context_effect_ci': dict(zip(METRICS, np.asarray(interval(boot_context)).T.tolist())),
        'injected_effect_ci': dict(zip(METRICS, np.asarray(interval(boot_injection)).T.tolist())),
        'context_matches_intended': denominator * expected > 0,
        'injected_matches_intended': numerator * expected > 0,
        'transfer_ratio': ratio, 'transfer_ratio_ci': ratio_ci,
        'transfer_ratio_ci_reason': ratio_reason,
        'bootstrap': {'unit': 'question', 'draws': 10000, 'seed': seed},
        'formal_e1_pass': False, 'behavior_validated': False,
        'limitation': '题目区间不代表跨经历或家族不确定性；正迁移比不保证预期方向正确；未核验采集来源或能力',
    }
