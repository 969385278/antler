"""C11：稳定性复测不泄漏参考留出题，不按结果择题。"""
from copy import deepcopy

import pytest

from antler.memory.e1_probe_repeat import probe_repeat_plan


def source():
    pool = [f'Question {i}?' for i in range(100)]
    return pool, {'source': 'trait_extract', 'fit': pool[:20], 'evaluation': pool[20:30]}


def test_fixed_disjoint_probes_and_common_heldout():
    pool, reference = source(); original = deepcopy(reference)
    result = probe_repeat_plan(pool, reference)
    assert reference == original
    assert result == probe_repeat_plan(list(reversed(pool)), reference)
    assert len(result['fit']) == len(set(result['fit'])) == 20
    assert not set(result['fit']) & set(pool[:30])
    assert result['evaluation'] == reference['evaluation']
    assert result['reference_fit'] == reference['fit']
    assert not result['formal_e1_pass']


@pytest.mark.parametrize('fault', ['duplicate_pool', 'overlap', 'outside', 'source', 'short', 'empty'])
def test_unsafe_or_incomplete_probe_sets_rejected(fault):
    pool, reference = source()
    if fault == 'duplicate_pool': pool[-1] = '  QUESTION 0? '
    elif fault == 'overlap': reference['evaluation'][0] = reference['fit'][0].upper()
    elif fault == 'outside': reference['fit'][0] = 'Unknown evaluation question'
    elif fault == 'source': reference['source'] = 'trait_eval'
    elif fault == 'short': pool = pool[:49]
    else: pool[-1] = ''
    with pytest.raises(ValueError): probe_repeat_plan(pool, reference)
