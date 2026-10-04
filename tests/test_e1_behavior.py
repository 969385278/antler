"""C11：行为迁移的五记忆配对、分母与缺失数据测试；无模型下载。"""
from copy import deepcopy

import pytest

from antler.eval.e1_behavior import summarize_behavior


def report(score, *, pole='high', repeats=1):
    return {'dim': 'C', 'pole': pole, 'split': 'test', 'samples': [
        {'question_index': q, 'sample_index': s, 'q': f'题目{q}',
         'trait': score if isinstance(score, (int, float)) else score[q - 10],
         'coherence': 90., 'task': 85.}
        for q in range(10, 20) for s in range(repeats)]}


def test_all_five_memories_change_denominator_and_injection():
    base = [report(v) for v in (10, 20, 30, 40, 50)]
    injection = [report(v) for v in (20, 30, 40, 50, 60)]
    r = summarize_behavior(base, report(50), injection, intended_sign=1)
    assert r['neutral']['trait'] == 30
    assert r['transfer_ratio'] == .5
    assert r['transfer_ratio_ci'] == [.5, .5]
    assert r['context_effect']['trait'] == 20
    assert r['injected']['coherence'] == 90
    assert not r['formal_e1_pass'] and not r['behavior_validated']


def test_both_wrong_sign_still_positive_ratio_is_disclosed():
    r = summarize_behavior([report(50)] * 5, report(30), [report(40)] * 5, intended_sign=1)
    assert r['transfer_ratio'] == .5
    assert not r['context_matches_intended'] and not r['injected_matches_intended']


def test_low_pole_uses_own_score_direction():
    r = summarize_behavior([report(30, pole='low')] * 5, report(50, pole='low'),
                           [report(40, pole='low')] * 5, intended_sign=-1)
    assert r['context_matches_intended'] and r['injected_matches_intended']


def test_zero_context_effect_kept_undefined():
    r = summarize_behavior([report(50)] * 5, report(50), [report(60)] * 5, intended_sign=1)
    assert r['transfer_ratio'] is None and r['transfer_ratio_ci'] is None
    assert r['injected_effect']['trait'] == 10


def test_uncertain_denominator_does_not_yield_finite_ratio_interval():
    r = summarize_behavior([report(50)] * 5, report([40] * 4 + [60] * 6),
                           [report(55)] * 5, intended_sign=1)
    assert r['transfer_ratio'] == 2.5
    assert r['transfer_ratio_ci'] is None
    assert r['context_effect_ci']['trait'][0] < 0 < r['context_effect_ci']['trait'][1]


def test_opposite_effect_and_overshoot_are_not_clipped():
    for score, expected in [(40, -1), (80, 3)]:
        r = summarize_behavior([report(50)] * 5, report(60), [report(score)] * 5, intended_sign=1)
        assert r['transfer_ratio'] == expected


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True, -1, 101])
def test_invalid_score_not_silently_dropped(value):
    base = [report(30) for _ in range(5)]
    base[4]['samples'][0]['task'] = value
    with pytest.raises(ValueError):
        summarize_behavior(base, report(50), [report(40)] * 5, intended_sign=1)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'question', 'calib', 'pole', 'sample'])
def test_pairing_corruption_rejected(mutation):
    base = [report(30) for _ in range(5)]
    r = base[0]
    if mutation == 'missing': r['samples'].pop()
    elif mutation == 'duplicate': r['samples'].append(deepcopy(r['samples'][0]))
    elif mutation == 'question': r['samples'][0]['q'] = '不同题目'
    elif mutation == 'calib': r['split'] = 'calib'
    elif mutation == 'pole': r['pole'] = 'low'
    else: r['samples'][0]['sample_index'] = 2
    with pytest.raises(ValueError):
        summarize_behavior(base, report(50), [report(40)] * 5, intended_sign=1)


def test_repeated_identical_samples_do_not_narrow_question_interval():
    def run(n):
        return summarize_behavior([report(30, repeats=n)] * 5,
                                  report(list(range(40, 50)), repeats=n),
                                  [report(list(range(35, 45)), repeats=n)] * 5,
                                  intended_sign=1)
    first, repeated = run(1), run(3)
    assert first['transfer_ratio_ci'] == repeated['transfer_ratio_ci']
    assert first['context_effect_ci'] == repeated['context_effect_ci']


def test_missing_neutral_memory_rejected():
    with pytest.raises(ValueError):
        summarize_behavior([report(30)] * 4, report(50), [report(40)] * 5, intended_sign=1)
