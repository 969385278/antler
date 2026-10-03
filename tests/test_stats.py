"""C8：用已知分布、完全配对和不平衡重复检验统计单位。"""
import pytest

from antler.eval.stats import bootstrap_mean, paired_bootstrap, wilson_interval, benjamini_hochberg


def test_repeating_one_question_does_not_inflate_weight():
    base = bootstrap_mean([0., 100.], ['a', 'b'])
    repeated = bootstrap_mean([0.] * 100 + [100.], ['a'] * 100 + ['b'])
    assert base == repeated and base['mean'] == 50. and base['n_questions'] == 2


def test_paired_offsets_have_zero_width_interval_and_missing_excluded():
    result = paired_bootstrap([1, 2, None, 80], ['a','b','c','d'], [12, 11, None], ['b','a','c'])
    assert result['mean'] == result['ci_low'] == result['ci_high'] == 10.
    assert result['n_unpaired_questions'] == 1 and result['n_questions'] == 2
    assert result['effect_size_dz'] is None


def test_wilson_known_reference_and_endpoints():
    result = wilson_interval(5,10)
    assert result['ci_low'] == pytest.approx(.23659309, abs=1e-7)
    assert result['ci_high'] == pytest.approx(.76340691, abs=1e-7)
    assert wilson_interval(0,10)['ci_low'] == pytest.approx(0.)
    assert wilson_interval(10,10)['ci_high'] == pytest.approx(1.)


def test_bh_reference_preserves_order_and_missing():
    result = benjamini_hochberg([.04, .01, None, .03, .20])
    assert result['adjusted_p'][0] == pytest.approx(.0533333333)
    assert result['adjusted_p'][1] == .04
    assert result['adjusted_p'][2] is None
    assert result['reject'] == [False, True, False, False, False]


def test_insufficient_or_invalid_inputs_fail():
    with pytest.raises(ValueError):
        bootstrap_mean([None], ['a'])
    with pytest.raises(ValueError):
        bootstrap_mean([1,2], ['a','a'])
    with pytest.raises(ValueError):
        paired_bootstrap([1,2], ['a','b'], [3,4], ['c','d'])
    with pytest.raises(ValueError):
        bootstrap_mean([float('nan'),1], ['a','b'])
