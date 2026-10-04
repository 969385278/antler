"""C11：拒绝部分样本、基线拼接和丢弃负解释率的汇总测试。"""
from copy import deepcopy

import pytest

from antler.eval.e1_summary import summarize


def dataset():
    def metric(v):
        return {'kl': v, 'per_probe': [{'probe': f'q{i}', 'kl': v} for i in range(10)]}
    rows = []
    for i in range(12):
        arms = {k: {'baseline': metric(1), 'final': metric(v), 'vector': [v]*8, 'coordinates': [1]*5}
                for k, v in [('personality', .8), ('random', .9), ('full', 2)]}
        rows.append({'id': f'p{i+1:02}', 'intended': {'O': 1, 'C': -1} if i < 4 else {'O': 1},
                     'arms': arms, 'A_eval': metric(.95), 'delta_A': [1]*5})
    return rows


def test_signed_results_and_label_counts():
    report = summarize(dataset())
    assert report['arms']['full']['explained']['mean'] == -1
    assert report['personality_minus_random_explained']['mean'] == pytest.approx(.1)
    assert report['semantic']['B']['hits'] == 12
    assert report['semantic']['B']['per_dimension']['C']['negative_labels'] == 4
    assert report['formal_e1_pass'] is False


def test_correct_sign_does_not_imply_specificity():
    rows = dataset()
    rows[0]['arms']['personality']['coordinates'] = [.1, -.1, 2, 0, 0]
    report = summarize(rows)
    ranks = report['semantic']['B']['per_episode'][0]['intended_absolute_ranks']
    assert ranks == {'O': 2, 'C': 2}
    assert report['semantic']['B']['hits'] == 13


def test_coordinate_agreement_keeps_opposite_orthogonal_and_undefined():
    rows = dataset()
    rows[0]['arms']['personality']['coordinates'] = [-1]*5
    rows[1]['delta_A'] = [1, 0, 0, 0, 0]
    rows[1]['arms']['personality']['coordinates'] = [0, 1, 0, 0, 0]
    rows[2]['delta_A'] = [0]*5
    report = summarize(rows)['A_B_agreement']
    values = [r['coordinate_cosine'] for r in report['per_episode']]
    assert values[:2] == pytest.approx([-1, 0])
    assert values[2] is None
    assert values[3:] == pytest.approx([1]*9)
    assert report['coordinate_cosine']['undefined'] == 1
    assert report['coordinate_cosine']['mean'] == pytest.approx(8/11)


def test_coordinate_agreement_is_scale_invariant_and_preserves_input():
    rows = dataset()
    rows[0]['delta_A'] = [1e300]*5
    rows[0]['arms']['personality']['coordinates'] = [1e-300]*5
    original = deepcopy(rows)
    report = summarize(rows)['A_B_agreement']
    assert report['coordinate_cosine']['mean'] == pytest.approx(1)
    assert rows == original


def test_coordinate_agreement_all_zero_is_not_perfect_agreement():
    rows = dataset()
    for row in rows:
        row['delta_A'] = [0]*5
    summary = summarize(rows)['A_B_agreement']['coordinate_cosine']
    assert summary['mean'] is None
    assert summary['episode_bootstrap_ci'] is None
    assert summary['undefined'] == 12


@pytest.mark.parametrize('change', ['partial', 'baseline', 'probe', 'nonfinite', 'mean'])
def test_invalid_comparisons_rejected(change):
    rows = deepcopy(dataset())
    if change == 'partial': rows.pop()
    elif change == 'baseline': rows[0]['arms']['random']['baseline']['kl'] = 2
    elif change == 'probe': rows[0]['arms']['random']['final']['per_probe'][0]['probe'] = 'other'
    elif change == 'mean': rows[0]['A_eval']['kl'] = 10
    else: rows[0]['arms']['full']['final']['kl'] = float('nan')
    with pytest.raises(ValueError):summarize(rows)
