"""C11：非正交几何、行为关联和无模型入口的 CPU 合成测试。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from antler.eval.e1_coordinate_audit import BUNDLES, analyze, validate_basis
from antler.traits.data import DIMS


HASH = 'a' * 64


@pytest.fixture
def inputs():
    u = torch.eye(5, 8, dtype=torch.float64)
    u[1] = .8 * u[0] + .6 * u[1]
    directions = {'dims': list(DIMS), 'U': u.repeat(17, 1, 1),
                  's': torch.full((17,), 2., dtype=torch.float64)}
    rows, behavior_rows = [], []
    for index, dim in enumerate(DIMS):
        for sign in (-1, 1):
            identity = f'{dim}_{sign}'
            delta = [0.] * 5
            delta[index] = sign * .2
            delta[0] += .5
            row = {'id': identity, 'text_sha256': str(index + 1) * 64,
                   'group_id': f'family_{index}', 'intended': {dim: sign}, 'delta': delta}
            rows.append(row)
            reports = []
            for pole in ('high', 'low'):
                effect = sign * (1 if pole == 'high' else -1) * 10.
                metrics = {}
                for metric, change in [('trait', effect), ('coherence', -2.), ('task', -1.)]:
                    metrics[metric] = {'context': 50. + change, 'neutral': 50., 'none': 50.,
                                       'context_minus_neutral': change,
                                       'question_bootstrap_ci': [change - 1., change + 1.]}
                reports.append({'id': identity, 'dim': dim, 'pole': pole,
                                'missing_or_invalid_rows': 0, 'status': 'descriptive', 'metrics': metrics})
            behavior_rows.append({key: deepcopy(row[key]) for key in ('id', 'text_sha256', 'intended')})
            behavior_rows[-1]['context_reports'] = reports
    settings = {'projection': 'dual', 'neutral_count': 5, 'steer_phase': 'all',
                'activation_average': 'all_fit_probes'}
    bundles = {name: {'method': name[0], 'layer': 16, 'basis_sha256': HASH, 'units': 's_layer',
                      'fit_settings': deepcopy(settings), 'probes': ['独立探针'], 'episodes': deepcopy(rows)}
               for name in BUNDLES}
    return bundles, {'episodes': behavior_rows}, directions


def compute(inputs):
    bundles, behavior, directions = inputs
    return analyze(bundles, behavior, directions, layer=16, basis_sha256=HASH)


def test_nonorthogonal_projection_can_reverse_a_coefficient(inputs):
    for bundle in inputs[0].values():
        bundle['episodes'][0]['delta'] = [-.2, 1., 0., 0., 0.]
    result = compute(inputs)
    row = result['bundles']['A_reference']['episodes'][0]
    assert row['coordinates']['dual_raw'][0] == -.2
    assert row['coordinates']['axis_raw'][0] == pytest.approx(.6)
    # 实际向量是 .6 e0 + .6 e1；不是坐标的普通 L2 范数。
    assert row['physical_norm_s'] == pytest.approx((.6 ** 2 + .6 ** 2) ** .5)
    assert row['physical_norm_absolute'] == pytest.approx(2 * row['physical_norm_s'])
    assert result['bundles']['A_reference']['axis_projection_scope'] == 'original_activation_difference_projection'
    assert result['bundles']['B_reference']['axis_projection_scope'] == 'fitted_injection_vector_projection'


def test_centering_and_pair_difference_do_not_create_twins(inputs):
    result = compute(inputs)
    bundle = result['bundles']['A_reference']
    assert bundle['mean_dual'] == pytest.approx([.5, 0., 0., 0., 0.])
    assert bundle['sum_physical_energy_s_squared'] == pytest.approx(
        bundle['common_mean_energy_s_squared'] + bundle['sum_centered_physical_energy_s_squared'])
    assert bundle['common_mean_energy_fraction'] == pytest.approx(2.5 / 2.9)
    assert bundle['sign_counts']['dual_raw']['correct'] == 9
    assert bundle['sign_counts']['dual_centered']['correct'] == 10
    for pair in bundle['pairs']:
        assert pair['same_family'] is True
        assert pair['controlled_outcome_twins'] is False
        assert pair['plus_minus_dual_centered'] == pytest.approx(pair['plus_minus_dual'])
        assert pair['physical_difference_norm_s'] == pytest.approx(.4)
        assert pair['behavior_effect_differences']['high']['trait']['difference_ci'] is None


def test_low_pole_orients_trait_and_preserves_quality_and_uncertainty(inputs):
    reports = inputs[1]['episodes'][0]['context_reports']
    reports[0]['metrics']['trait']['question_bootstrap_ci'] = [-20., 1.]
    result = compute(inputs)
    reports = result['bundles']['A_reference']['episodes'][0]['context_reports']
    assert reports[0]['trait_high_axis']['ci_state'] == 'uncertain'
    assert reports[1]['metrics']['trait']['context_minus_neutral'] == 10.
    assert reports[1]['trait_high_axis']['effect'] == -10.
    assert reports[1]['trait_high_axis']['question_bootstrap_ci'] == [-11., -9.]
    for report in reports:
        assert report['metrics']['coherence']['context_minus_neutral'] == -2.
        assert report['metrics']['task']['context_minus_neutral'] == -1.
    assert result['bundles']['A_reference']['behavior_sign_counts']['dual_raw']['high']['ci_uncertain'] == 1


def test_all_four_bundles_and_ten_episodes_preserved_without_mutation(inputs):
    before = deepcopy(inputs[:2])
    result = compute(inputs)
    assert inputs[:2] == before
    assert result['n_episodes'] == 10 and result['n_groups'] == 5
    assert set(result['bundles']) == set(BUNDLES)
    assert not result['formal_e1_pass'] and not result['behavior_validated']
    assert not result['injection_performed']
    for bundle in result['bundles'].values():
        assert len(bundle['episodes']) == 10 and len(bundle['pairs']) == 5


def test_zero_vectors_report_undefined_energy_not_success(inputs):
    for bundle in inputs[0].values():
        for row in bundle['episodes']:
            row['delta'] = [0.] * 5
    result = compute(inputs)
    for bundle in result['bundles'].values():
        assert bundle['common_mean_energy_fraction'] is None
        assert bundle['sign_counts']['dual_raw']['correct'] == 0
        assert all(pair['raw_pair_physical_cosine'] is None for pair in bundle['pairs'])


@pytest.mark.parametrize('change', [
    lambda b: b.pop('B_repeat'),
    lambda b: b['A_reference'].update(layer=15),
    lambda b: b['A_reference'].update(units='raw'),
    lambda b: b['A_reference'].update(basis_sha256='b' * 64),
    lambda b: b['A_reference'].update(dims=list(reversed(DIMS))),
    lambda b: b['A_reference']['episodes'].pop(),
    lambda b: b['A_reference']['episodes'].__setitem__(0, deepcopy(b['A_reference']['episodes'][1])),
    lambda b: b['A_reference']['episodes'][0].update(delta=[float('nan')] * 5),
    lambda b: b['A_reference']['episodes'][0].update(delta=[float('inf')] * 5),
    lambda b: b['A_reference']['episodes'][0].update(delta=[False] * 5),
    lambda b: b['A_reference']['episodes'][0].update(intended={'O': True}),
    lambda b: b['B_repeat']['episodes'][0].update(text_sha256='b' * 64),
    lambda b: b['B_repeat']['episodes'][0].update(group_id='another_family'),
    lambda b: b['A_reference']['probes'].append('独立探针'),
    lambda b: b['B_repeat']['fit_settings'].update(neutral_count=1),
])
def test_coordinate_identity_and_metadata_errors_rejected(inputs, change):
    change(inputs[0])
    with pytest.raises(ValueError):
        compute(inputs)


@pytest.mark.parametrize('change', [
    lambda b: b['episodes'].pop(),
    lambda b: b['episodes'][0].update(text_sha256='b' * 64),
    lambda b: b['episodes'][0].update(intended={'O': 1}),
    lambda b: b['episodes'][0]['context_reports'][0].update(missing_or_invalid_rows=1),
    lambda b: b['episodes'][0]['context_reports'][0].update(pole='low'),
    lambda b: b['episodes'][0]['context_reports'][0]['metrics'].pop('task'),
    lambda b: b['episodes'][0]['context_reports'][0]['metrics']['trait'].update(context_minus_neutral=1.),
    lambda b: b['episodes'][0]['context_reports'][0]['metrics']['trait'].update(question_bootstrap_ci=[2., -2.]),
    lambda b: b['episodes'][0]['context_reports'][0]['metrics']['task'].update(question_bootstrap_ci=[float('nan'), 1.]),
])
def test_behavior_conflicts_and_missing_quality_rejected(inputs, change):
    change(inputs[1])
    with pytest.raises(ValueError):
        compute(inputs)


def test_rank_scale_and_expected_identity_guard(inputs):
    directions = inputs[2]
    directions['U'][:, 1] = directions['U'][:, 0]
    with pytest.raises(ValueError, match='满秩'):
        validate_basis(directions, 16)
    directions['s'][16] = 0.
    with pytest.raises(ValueError, match='不为正'):
        validate_basis(directions, 16)
    fresh = torch.eye(5, 8, dtype=torch.float64)
    directions['U'] = fresh.repeat(17, 1, 1)
    directions['s'][16] = 2.
    with pytest.raises(ValueError, match='全部十段'):
        analyze(*inputs[:2], directions, layer=16, basis_sha256=HASH, expected_ids=['不存在'])


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e1_coordinate_behavior_audit')


def stage_files(module, inputs, tmp_path, monkeypatch):
    bundles, behavior, directions = inputs
    coordinate_dir = tmp_path / 'coordinates'
    coordinate_dir.mkdir()
    for name, bundle in bundles.items():
        bundle['basis_sha256'] = 'placeholder'
        (coordinate_dir / f'{name}.json').write_text(json.dumps(bundle), encoding='utf-8')
    direction_path = tmp_path / 'directions.pt'
    torch.save(directions, direction_path)
    basis_hash = module.sha256(direction_path)
    for name, bundle in bundles.items():
        bundle['basis_sha256'] = basis_hash
        (coordinate_dir / f'{name}.json').write_text(json.dumps(bundle), encoding='utf-8')
    behavior_path = tmp_path / 'behavior.json'
    behavior_path.write_text(json.dumps(behavior), encoding='utf-8')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(f'out_dir: "{tmp_path.as_posix()}/out"\nuse_judge: true\n', encoding='utf-8')
    monkeypatch.setattr(module, 'DIRECTIONS_SHA256', basis_hash)
    hashes = {name: module.sha256(coordinate_dir / f'{name}.json') for name in BUNDLES}
    hashes['behavior'] = module.sha256(behavior_path)
    monkeypatch.setattr(module, 'INPUT_SHA256', hashes)
    monkeypatch.setattr(module, 'EXPECTED_IDS', tuple(row['id'] for row in bundles['A_reference']['episodes']))
    common = importlib.import_module('_common')
    def forbidden(*args, **kwargs):
        pytest.fail('离线入口不得加载模型或构造评委')
    monkeypatch.setattr(common, 'load_model', forbidden)
    from antler.eval.judge import Judge
    monkeypatch.setattr(Judge, '__init__', forbidden)
    return SimpleNamespace(config=str(config_path), directions=str(direction_path),
                           coordinate_dir=str(coordinate_dir), behavior=str(behavior_path))


def test_real_setup_without_model_and_hash_failure_archive(module, inputs, tmp_path, monkeypatch):
    args = stage_files(module, inputs, tmp_path, monkeypatch)
    out = module.run(args)
    meta = json.loads((out / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed' and not meta['model_loaded']
    assert meta['model_forward_calls'] == meta['generation_calls'] == meta['judge_calls'] == 0
    assert meta['device'] == 'cpu' and meta['gpu'] is None and meta['judge'] is None
    hashes = json.loads((out / 'artifact_hashes.json').read_text(encoding='utf-8'))
    assert hashes['analysis.json'] == module.sha256(out / 'analysis.json')
    assert hashes['run_meta.json'] == module.sha256(out / 'run_meta.json')
    Path(args.behavior).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='哈希'):
        module.run(args)
    metas = [json.loads(p.read_text(encoding='utf-8')) for p in out.parent.glob('*/run_meta.json')]
    assert sorted(m['status'] for m in metas) == ['completed', 'failed']


def test_input_drift_during_analysis_is_detected(module, inputs, tmp_path, monkeypatch):
    args = stage_files(module, inputs, tmp_path, monkeypatch)
    original = module.analyze
    def drift(*args_, **kwargs):
        result = original(*args_, **kwargs)
        Path(args.behavior).write_text('{}', encoding='utf-8')
        return result
    monkeypatch.setattr(module, 'analyze', drift)
    with pytest.raises(ValueError, match='发生变化'):
        module.run(args)
    run_dir = next((tmp_path / 'out/e1_coordinate_behavior_audit').iterdir())
    assert not (run_dir / 'analysis.json').exists()
    assert json.loads((run_dir / 'run_meta.json').read_text(encoding='utf-8'))['status'] == 'failed'


def test_duplicate_json_keys_are_rejected(module, tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"layer": 16, "layer": 15}', encoding='utf-8')
    with pytest.raises(ValueError, match='重复键'):
        module._read(path)
