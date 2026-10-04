"""C11：dev固定平衡抽样，不使用test或按模型结果挑选。"""
import pytest
from copy import deepcopy

from antler.memory.d1_plan import make_plan
from antler.memory.e1_dev_design import select_dev, learning_rates
from test_episode_data import candidate


def rows():
    return [{**candidate(s['id'],s['intended']), **s} for s in make_plan()['slots']]


def test_dev_selection_is_order_independent_balanced_and_preserves_inputs():
    data = rows(); before = deepcopy(data)
    a = select_dev(data)
    assert a == select_dev(data[::-1]) and data == before
    assert len(a) == 10 and all(r['split'] == 'dev' for r in a)
    assert {tuple(r['intended'].items()) for r in a} == {((d,s),) for d in 'OCEAN' for s in (-1,1)}


def test_incomplete_dev_cannot_be_filled_from_test():
    data = rows()
    remove = next(r for r in data if r['split']=='dev' and r['intended']=={'O':1})
    data.remove(remove)
    with pytest.raises(ValueError): select_dev(data)


def test_learning_rates_match_the_registered_geometric_reference():
    rates = learning_rates(2560)
    assert rates['personality'] == rates['random'] == [.005,.02,.08]
    assert rates['full'][1] == pytest.approx(.0008838834764831844)
    with pytest.raises(ValueError): learning_rates(0)
