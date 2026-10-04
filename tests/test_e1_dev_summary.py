"""C11：防止不完整网格、探针泄漏及按其他指标事后选参。"""
from copy import deepcopy

import pytest

from antler.memory.e1_dev_design import learning_rates
from antler.memory.e1_dev_summary import summarize_grid


def fixture():
    episodes = [{'id': f'{d}{s}', 'intended': {d: s}, 'split': 'dev'}
                for d in 'OCEAN' for s in (-1, 1)]
    rows = []
    for episode in episodes:
        arms = {}
        for space, rates in learning_rates(2560).items():
            for i, rate in enumerate(rates):
                arms[f'{space}_lr{i}'] = {
                    'baseline': {'kl': 1.0}, 'final': {'kl': [.4, .2, .2][i]},
                    'steer_phase': 'all', 'behavior_validated': False,
                    'fit_probes': [f'fit {j}' for j in range(20)],
                    'evaluation_probes': [f'eval {j}' for j in range(10)],
                    'history': [{'step': j, 'fit_minibatch_kl_before_update': .1} for j in range(60)],
                    'diagnostics': {'optimizer': {'name': 'Adam', 'lr': rate},
                        'fit_baseline': {'kl': 1.}, 'fit_initial': {'kl': .5},
                        'fit_final': {'kl': [.01, .4, .05][i]}, 'heldout_initial': {'kl': .7},
                        'step_metrics': [{'step': j, 'vector_update_norm': .01} for j in range(60)]}}
        rows.append({'id': episode['id'], 'intended': episode['intended'], 'arms': arms})
    return {'episodes': rows, 'preflight': False, 'formal_e1_pass': False}, {
        'scope': 'dev_optimization_only', 'episodes': episodes}


def test_selects_heldout_mean_with_lower_rate_tiebreak_not_fit_score():
    result, manifest = fixture()
    before = deepcopy(result)
    report = summarize_grid(result, manifest, hidden=2560)
    assert report['n_fits'] == 90 and not report['formal_e1_pass']
    assert all(v['selected_arm'] == f'{k}_lr1' for k, v in report['spaces'].items())
    assert result == before


@pytest.mark.parametrize('problem', ['episode', 'arm', 'steps', 'nan', 'overlap', 'baseline', 'preflight', 'test', 'lr'])
def test_rejects_invalid_evidence(problem):
    result, manifest = fixture()
    arm = result['episodes'][0]['arms']['personality_lr0']
    if problem == 'episode': result['episodes'].pop()
    if problem == 'arm': result['episodes'][0]['arms'].pop('random_lr2')
    if problem == 'steps': arm['history'].pop()
    if problem == 'nan': arm['final']['kl'] = float('nan')
    if problem == 'overlap': arm['evaluation_probes'][0] = ' FIT   0 '
    if problem == 'baseline': arm['baseline']['kl'] = 2.
    if problem == 'preflight': result['preflight'] = True
    if problem == 'test': manifest['episodes'][0]['split'] = 'test'
    if problem == 'lr': arm['diagnostics']['optimizer']['lr'] = .9
    with pytest.raises(ValueError): summarize_grid(result, manifest, hidden=2560)
