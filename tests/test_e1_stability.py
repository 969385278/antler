"""C11：重复测量不能用同一探针、A/B混比或零向量伪造稳定性。"""
from copy import deepcopy
import pytest
from antler.eval.e1_stability import compare_probe_replicates


def pair():
    a = {'method': 'A', 'layer': 16, 'basis_sha256': 'basis', 'units': 's_layer', 'fit_settings': {},
         'probes': ['q1', 'q2'], 'episodes': [
             {'id': 'a', 'text_sha256': 'text-a', 'group_id': 'g', 'intended': {'O': 1}, 'delta': [1, 2, 3, 4, 5]},
             {'id': 'b', 'text_sha256': 'text-b', 'group_id': 'g', 'intended': {'C': -1}, 'delta': [0, -1, 0, 0, 0]}]}
    b = deepcopy(a); b['probes'] = ['q3', 'q4']
    return a, b


def test_same_direction_different_amplitude_and_group_count():
    a, b = pair(); b['episodes'][0]['delta'] = [2, 4, 6, 8, 10]
    result = compare_probe_replicates(a, b)
    assert result['n_groups'] == 1 and result['n_episodes'] == 2
    assert result['cosine_mean'] == pytest.approx(1)
    assert result['per_episode'][0]['norm_ratio'] == pytest.approx(2)
    assert result['per_episode'][0]['relative_l2'] == pytest.approx(1)
    assert result['target_signs']['O']['both_correct'] == 1
    assert result['formal_e1_pass'] is False


def test_consistently_wrong_sign_is_not_correct_semantics():
    a, b = pair()
    a['episodes'][0]['delta'][0] = b['episodes'][0]['delta'][0] = -1
    counts = compare_probe_replicates(a, b)['target_signs']['O']
    assert counts == {'target_count': 1, 'sign_agree': 1, 'both_correct': 0}


def test_zero_vector_has_undefined_cosine_not_perfect_agreement():
    a, b = pair()
    a['episodes'][0]['delta'] = b['episodes'][0]['delta'] = [0]*5
    result = compare_probe_replicates(a, b)
    assert result['cosine_undefined'] == 1
    assert result['per_episode'][0]['cosine'] is None
    assert result['per_episode'][0]['norm_ratio'] is None


@pytest.mark.parametrize('problem', ['method', 'basis_sha256', 'layer', 'units', 'fit_settings', 'overlap', 'missing', 'text', 'nan'])
def test_rejects_noncomparable_repeats(problem):
    a, b = pair()
    if problem in ('method', 'basis_sha256', 'layer', 'units', 'fit_settings'): b[problem] = 'changed'
    if problem == 'overlap': b['probes'][0] = ' Q1 '
    if problem == 'missing': b['episodes'].pop()
    if problem == 'text': b['episodes'][0]['text_sha256'] = 'other'
    if problem == 'nan': b['episodes'][0]['delta'][0] = float('nan')
    with pytest.raises(ValueError): compare_probe_replicates(a, b)
