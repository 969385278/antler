"""C11：在看到Qwen结果前固定平衡dev子集与各参数化学习率。"""
import hashlib
import math

from .episode_data import validate_candidates, validate_split_plan
from .d1_plan import make_plan
from ..traits.data import DIMS


def select_dev(selected):
    dev = [r for r in selected if r.get('split') == 'dev' and len(r['intended']) == 1]
    validate_candidates(dev, allow_grouped=True)
    validate_split_plan(dev, make_plan())
    chosen = []
    for dim in DIMS:
        for sign in (-1, 1):
            group = [r for r in dev if r['intended'] == {dim: sign}]
            if len(group) != 4:
                raise ValueError('dev每个指定方向必须恰有4条候选')
            chosen.append(min(group, key=lambda r: hashlib.sha256(('E1-dev-grid-0:'+r['id']).encode()).hexdigest()))
    return chosen


def learning_rates(hidden):
    if type(hidden) is not int or hidden < 5:
        raise ValueError('隐藏维数非法')
    rates = [.005, .02, .08]
    return {'personality': rates, 'random': rates.copy(),
            'full': [r*math.sqrt(5/hidden) for r in rates]}
