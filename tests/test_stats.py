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


@pytest.mark.parametrize('offset, expected', [(0, -10.0), (-.03, -10.001), (.03, -9.999)])
def test_quality_boundary_from_repeated_question_scores(offset, expected):
    """真实评分结构的边界及两侧；不能用容差把真实超限归为通过。"""
    baseline = [95,85,85,90,85,95,95,95,85,95,95,85,95,90,85,
                95,95,90,95,95,95,85,95,95,95,95,95,95,90,95]
    treated = [60,85,85,85,85,85,95,60,90,95,85,50,85,90,75,
               85,95,85,85,90,85,85,60,90,60,95,95,60,90,90]
    treated[0] += offset
    ids = [i // 3 for i in range(30)]
    result = paired_bootstrap(baseline, ids, treated, ids)
    assert result['mean'] == expected
    assert (result['mean'] >= -10) == (offset >= 0)


def test_exact_paired_mean_keeps_question_weights_with_missing_and_unequal_counts():
    result = paired_bootstrap([0,0,0,30,None,99], ['a','a','a','b','c','d'],
                              [30,0,0,None], ['a','b','b','c'])
    assert result['mean'] == 0  # 每题变化+30/-30等权，不能按回答条数加权。
    assert result['n_questions'] == 2 and result['n_unpaired_questions'] == 1
