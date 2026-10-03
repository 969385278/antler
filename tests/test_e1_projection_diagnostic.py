"""C7：固定探索入口的CPU离线测试，只使用合成输入和随机微型模型。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from antler.core.activations import last_prompt_acts
from antler.core.model import get_layers
from antler.memory.write import MEMORY_TEMPLATE, NEUTRAL_MEMORIES, projection_coordinates
from antler.traits.data import DIMS
from test_core import D, L, model
from test_generate import Tokenizer


INTENDED = [
    {'C': 1, 'O': -1, 'N': 1}, {'E': 1}, {'O': 1}, {'A': 1},
    {'C': 1, 'N': 1}, {'N': -1}, {'E': -1}, {'O': -1, 'C': 1},
    {'A': 1}, {'N': 1}, {'A': -1}, {'C': 1},
]


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e1_projection_diagnostic')


def save_inputs(case, monkeypatch):
    torch.save(case.directions, case.directions_path)
    case.experiences_path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in case.episodes),
        encoding='utf-8', newline='\n')
    monkeypatch.setattr(case.module, 'DIRECTIONS_SHA256', case.module.sha256(case.directions_path))
    monkeypatch.setattr(case.module, 'EXPERIENCES_SHA256', case.module.sha256(case.experiences_path))


@pytest.fixture
def case(tmp_path, diagnostic, monkeypatch):
    u = torch.eye(5, D)
    u[1, 0], u[1, 1] = .6, .8
    directions = {'U': u.repeat(L, 1, 1), 's': torch.full((L,), 2.), 'dims': list(DIMS)}
    episodes = [{'id': f'p{i+1:02d}', 'text': f'合成经历{i+1}', 'intended': deepcopy(labels)}
                for i, labels in enumerate(INTENDED)]
    model_dir = tmp_path / 'local_model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{}\n', encoding='utf-8')
    config = tmp_path / 'config.yaml'
    config.write_text('model: unused\n', encoding='utf-8')
    monkeypatch.setattr(diagnostic, 'LAYER', 2)
    args = SimpleNamespace(config=str(config), directions=str(tmp_path / 'directions.pt'),
                           experiences=str(tmp_path / 'experiences.jsonl'))
    result = SimpleNamespace(root=tmp_path, module=diagnostic, directions=directions,
        directions_path=Path(args.directions), experiences_path=Path(args.experiences),
        episodes=episodes, model_dir=model_dir, config=config, args=args,
        probes=[f'{i+4} 3' for i in range(30)])
    save_inputs(result, monkeypatch)
    return result


def test_load_inputs_keeps_all_episodes_and_full_shapes(case):
    u, s, episodes = case.module.load_inputs(case.directions_path, case.experiences_path)
    assert u.shape == (L, 5, D) and s.shape == (L,)
    assert torch.equal(u, case.directions['U']) and torch.equal(s, case.directions['s'])
    assert episodes == case.episodes and len(episodes) == 12
    assert sum(len(ep['intended']) for ep in episodes) == 16


@pytest.mark.parametrize('kind', ['directions', 'experiences'])
def test_hash_mismatch_rejected(case, kind):
    path = case.directions_path if kind == 'directions' else case.experiences_path
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        case.module.load_inputs(case.directions_path, case.experiences_path)


@pytest.mark.parametrize('change', ['dims', 'missing_s', 'u_rank', 'u_hidden', 'u_layer',
    'u_integer', 'u_nan', 'u_nonunit', 'u_singular', 's_shape', 's_zero', 's_nan'])
def test_direction_shape_scale_and_basis_checks(case, monkeypatch, change):
    data = case.directions
    if change == 'dims': data['dims'] = list(reversed(DIMS))
    elif change == 'missing_s': data.pop('s')
    elif change == 'u_rank': data['U'] = data['U'][0]
    elif change == 'u_hidden': data['U'] = data['U'][:, :, :4]
    elif change == 'u_layer': data['U'] = data['U'][:2]
    elif change == 'u_integer': data['U'] = data['U'].long()
    elif change == 'u_nan': data['U'][0, 0, 0] = float('nan')
    elif change == 'u_nonunit': data['U'][0, 0] *= 2
    elif change == 'u_singular': data['U'][:, 1] = data['U'][:, 0]
    elif change == 's_shape': data['s'] = torch.ones(L, 1)
    elif change == 's_zero': data['s'][0] = 0
    else: data['s'][0] = float('nan')
    save_inputs(case, monkeypatch)
    with pytest.raises(ValueError):
        case.module.load_inputs(case.directions_path, case.experiences_path)


@pytest.mark.parametrize('change', ['missing_episode', 'duplicate_id', 'empty_text', 'missing_label',
    'bad_dimension', 'zero_sign', 'bool_sign', 'float_sign'])
def test_pilot_identity_and_16_labels_checked(case, monkeypatch, change):
    if change == 'missing_episode': case.episodes.pop()
    elif change == 'duplicate_id': case.episodes[-1]['id'] = case.episodes[0]['id']
    elif change == 'empty_text': case.episodes[0]['text'] = ''
    elif change == 'missing_label': case.episodes[0]['intended'].pop('N')
    elif change == 'bad_dimension': case.episodes[0]['intended']['X'] = case.episodes[0]['intended'].pop('N')
    elif change == 'zero_sign': case.episodes[0]['intended']['N'] = 0
    elif change == 'bool_sign': case.episodes[0]['intended']['N'] = True
    else: case.episodes[0]['intended']['N'] = 1.0
    save_inputs(case, monkeypatch)
    with pytest.raises(ValueError):
        case.module.load_inputs(case.directions_path, case.experiences_path)


@pytest.mark.parametrize('kind', ['missing', 'count', 'layers', 'hidden', 'integer', 'nan', 'inf'])
def test_activation_validation_rejects_incomplete_and_nonfinite(diagnostic, kind):
    tensor = torch.zeros(30, 1, D)
    if kind == 'missing': tensor = None
    elif kind == 'count': tensor = tensor[:-1]
    elif kind == 'layers': tensor = torch.zeros(30, 2, D)
    elif kind == 'hidden': tensor = tensor[:, :, :-1]
    elif kind == 'integer': tensor = tensor.long()
    elif kind == 'nan': tensor[0, 0, 0] = float('nan')
    else: tensor[0, 0, 0] = float('inf')
    with pytest.raises(ValueError):
        diagnostic.validate_activations(tensor, 30, D)


def test_activation_float32_conversion_overflow_is_rejected(diagnostic):
    tensor = torch.full((30, 1, D), 1e300, dtype=torch.float64)
    assert torch.isfinite(tensor).all()
    with pytest.raises(ValueError, match='float32'):
        diagnostic.validate_activations(tensor, 30, D)


class NumericChatTokenizer(Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        system = messages[0]['content'] if messages[0]['role'] == 'system' else ''
        marker = 4 + sum(map(ord, system)) % 40
        return f'{marker} ' + messages[-1]['content']


def test_real_tiny_cpu_forward_validates_prompt_activations(diagnostic, model):
    tok = NumericChatTokenizer()
    prompts = [tok.apply_chat_template([
        {'role': 'system', 'content': MEMORY_TEMPLATE.format(experience=NEUTRAL_MEMORIES[0])},
        {'role': 'user', 'content': f'{4+i} 3'}]) for i in range(30)]
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    acts = last_prompt_acts(model, tok, prompts, layers=[2], batch_size=16)
    diagnostic.validate_activations(acts, 30, D)
    assert acts.shape == (30, 1, D) and torch.isfinite(acts).all()
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, before[name])
               for name, p in model.named_parameters())
    assert all(not layer._forward_hooks for layer in get_layers(model))


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(D), requires_grad=False)
        self.config = SimpleNamespace(num_hidden_layers=L, hidden_size=D,
                                      to_dict=lambda: {'num_hidden_layers': L, 'hidden_size': D})


@pytest.fixture
def mocked_run(case, monkeypatch):
    module = case.module
    kit = case.root / 'kit'
    (kit / 'trait_extract').mkdir(parents=True)
    for pole in ('inventive', 'dependable', 'outgoing', 'compassionate', 'nervous'):
        (kit / 'trait_extract' / f'{pole}.json').write_text('{}\n', encoding='utf-8')
    monkeypatch.setattr(module, 'KIT', kit)
    state = SimpleNamespace(calls=[], projection_calls=[], model_loads=[], setup_calls=[],
                            model=FrozenModel(), changes=[])
    cfg = {'out_path': case.root / 'runs', 'model': str(case.model_dir),
           'dtype': 'float32', 'device': 'cpu', 'batch_size': 16, 'api_key': 'SECRET_TEST_TOKEN'}

    def setup(args, need_model, need_judge):
        state.setup_calls.append((need_model, need_judge))
        assert need_model is False and need_judge is False
        return cfg, None, None, None

    def load_model(*args, **kwargs):
        state.model_loads.append((args, kwargs))
        return state.model, NumericChatTokenizer()

    def probes(n, seed):
        assert n == 30 and seed == 0
        return list(case.probes)

    q_offsets = torch.arange(30, dtype=torch.float32).reshape(30, 1, 1) / 100
    for i in range(12):
        z = torch.tensor([.05 * (i - 5), .2, -.1, .15 * (-1 if i % 2 else 1), i * .025])
        state.changes.append(2 * (z @ case.directions['U'][2]))

    def acts(model, tok, prompts, *, layers, batch_size=16):
        assert model is state.model and layers == [2]
        assert batch_size == 16 and len(prompts) == 30
        index = len(state.calls)
        state.calls.append(list(prompts))
        if index < 5:
            return torch.full((30, 1, D), float(index)) + q_offsets
        return torch.full((30, 1, D), 2.) + q_offsets + state.changes[index - 5].reshape(1, 1, D)

    def project(change, u, scale, *, projection='dual'):
        if len(state.calls) >= 6:
            state.projection_calls.append((len(state.calls) - 6, projection, change.detach().clone()))
        return projection_coordinates(change, u, scale, projection=projection)

    monkeypatch.setattr(module, 'setup', setup)
    monkeypatch.setattr(module, 'load_model', load_model)
    monkeypatch.setattr(module, 'probe_questions', probes)
    monkeypatch.setattr(module, 'chat_prompt', lambda tok, p, system: system + '\n探针=' + p)
    monkeypatch.setattr(module, 'last_prompt_acts', acts)
    monkeypatch.setattr(module, 'projection_coordinates', project)
    state.acts = acts
    return state


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def test_run_reuses_five_neutral_batches_and_same_delta_for_both_readouts(case, mocked_run):
    state = mocked_run
    out = case.module.run(case.args)
    assert out.is_relative_to(case.root / 'runs')
    assert state.setup_calls == [(False, False)] and len(state.model_loads) == 1
    assert len(state.calls) == 17 and sum(map(len, state.calls)) == 510
    for index, memory in enumerate(NEUTRAL_MEMORIES):
        assert state.calls[index] == [MEMORY_TEMPLATE.format(experience=memory) + '\n探针=' + p
                                      for p in case.probes]
    for index, episode in enumerate(case.episodes):
        assert state.calls[index + 5] == [MEMORY_TEMPLATE.format(experience=episode['text']) + '\n探针=' + p
                                          for p in case.probes]
        projections = [(method, change) for ep, method, change in state.projection_calls if ep == index]
        assert {method for method, _ in projections} == {'dual', 'dot'} and len(projections) == 2
        assert torch.equal(projections[0][1], projections[1][1])
        assert torch.allclose(projections[0][1], state.changes[index], atol=2e-6)
    assert len(state.projection_calls) == 24
    geometry = read_json(out / 'basis_geometry.json')
    basis = case.directions['U'][2].double()
    gram = basis @ basis.T
    assert torch.tensor(geometry['gram']).shape == (5, 5)
    assert torch.allclose(torch.tensor(geometry['gram'], dtype=torch.float64), gram)
    assert geometry['condition_number'] == pytest.approx(float(torch.linalg.cond(gram)))
    assert geometry['basis_condition_number'] == pytest.approx(float(torch.linalg.cond(basis)))
    assert geometry['rank'] == 5
    assert torch.allclose(torch.tensor(geometry['row_norms'], dtype=torch.float64), basis.norm(dim=1))
    assert geometry['directions_sha256'] == case.module.sha256(case.directions_path)
    rendered = read_json(out / 'chat_prompts.json')
    assert len(rendered['neutral']) == 5 and len(rendered['episodes']) == 12
    assert [row['index'] for row in rendered['neutral']] == list(range(5))
    assert [row['id'] for row in rendered['episodes']] == [ep['id'] for ep in case.episodes]
    assert [row['prompts'] for row in rendered['neutral'] + rendered['episodes']] == state.calls
    assert all(len(row['prompts']) == 30 for row in rendered['neutral'] + rendered['episodes'])
    neutral = torch.load(out / 'neutral_activations.pt', weights_only=True)
    assert neutral['layer'] == 2 and len(neutral['per_memory']) == 5
    assert all(t.shape == (30, D) for t in neutral['per_memory'])
    expected_neutral = 2 + torch.arange(30).reshape(30, 1) / 100
    assert torch.allclose(neutral['neutral_mean'], expected_neutral.expand(30, D))
    differences = torch.load(out / 'activation_differences.pt', weights_only=True)
    assert differences['episode_ids'] == [ep['id'] for ep in case.episodes]
    assert differences['mean_differences'].shape == (12, D)
    assert torch.allclose(differences['mean_differences'], torch.stack(state.changes), atol=2e-6)
    assert differences['neutral_activations_sha256'] == case.module.sha256(out / 'neutral_activations.pt')
    probes = read_json(out / 'probes.json')
    assert probes['questions'] == case.probes and probes['seed'] == 0
    assert probes['source_split'] == 'trait_extract' and len(probes['source_hashes']) == 5
    assert all(Path(path).parent.name == 'trait_extract' for path in probes['source_hashes'])
    results = read_json(out / 'results.json')
    assert results['n_episodes'] == 12 and results['n_labels'] == 16
    assert not results['formal_e1_pass']
    meta = read_json(out / 'run_meta.json')
    assert meta['status'] == 'completed' and meta['end_time']
    assert meta['model_loaded'] is True and meta['judge_called'] is False
    assert meta['generation_performed'] is False and meta['formal_e1_pass'] is False
    assert meta['prompt_count'] == 510 and meta['neutral_groups_completed'] == 5 and meta['episodes_completed'] == 12
    assert meta['model_weights_sha256_verified'] is False
    assert meta['prompt_hashes']['rendered'] == case.module.sha256(out / 'chat_prompts.json')
    assert rendered['probe_sha256'] == meta['prompt_hashes']['probes']
    assert not any('trait_eval' in path for path in meta['data_hashes'])
    assert 'SECRET_TEST_TOKEN' not in (out / 'run_meta.json').read_text(encoding='utf-8')
    assert all(not p.requires_grad and p.grad is None for p in state.model.parameters())
    manifest = read_json(out / 'artifact_hashes.json')
    assert all(case.module.sha256(out / name) == digest for name, digest in manifest.items())
    with zipfile.ZipFile(out / 'results.zip') as archive:
        assert set(manifest) | {'artifact_hashes.json'} <= set(archive.namelist())
        assert archive.read('run_meta.json') == (out / 'run_meta.json').read_bytes()


def test_analyze_reports_raw_centered_counts_and_label_balance(case):
    values = [torch.tensor([float(i - 5), 1., -1., 1., -1.]) for i in range(12)]
    coordinates = {'dual': values, 'dot': [v * 2 for v in values]}
    result = case.module.analyze(case.episodes, coordinates)
    assert result['n_episodes'] == 12 and result['n_labels'] == 16
    assert result['formal_e1_pass'] is False
    assert set(result['projections']) == {'dual', 'dot'}
    expected_common = torch.stack(values).mean(0)
    for name, multiplier in [('dual', 1), ('dot', 2)]:
        report = result['projections'][name]
        assert torch.allclose(torch.tensor(report['common_mode']), expected_common * multiplier)
        assert report['raw']['total_labels'] == report['centered']['total_labels'] == 16
        raw_hits, centered_hits = 0, 0
        for ep, value in zip(case.episodes, values):
            raw_hits += sum(float(value[DIMS.index(dim)]) * sign > 0 for dim, sign in ep['intended'].items())
            centered_hits += sum(float((value - expected_common)[DIMS.index(dim)]) * sign > 0
                                 for dim, sign in ep['intended'].items())
        assert report['raw']['hits'] == raw_hits and report['centered']['hits'] == centered_hits
        c = report['raw']['per_dimension']['C']
        assert c['total_labels'] == c['positive_labels'] == 4 and c['negative_labels'] == 0
        assert len(report['per_episode']) == 12
        for ep, point, value in zip(case.episodes, report['per_episode'], values):
            assert point['id'] == ep['id'] and point['intended'] == ep['intended']
            assert point['n_labels'] == len(ep['intended'])
            assert torch.allclose(torch.tensor(point['raw']), value * multiplier)
            assert torch.allclose(torch.tensor(point['centered']), (value - expected_common) * multiplier)


@pytest.mark.parametrize('failure', ['hash', 'shape', 'missing_model_config', 'trainable_model',
                                     'missing_acts', 'nan_acts', 'source_changed'])
def test_failed_run_stops_and_retains_partial_outputs(case, mocked_run, monkeypatch, failure):
    state, module = mocked_run, case.module
    if failure == 'hash':
        case.experiences_path.write_bytes(case.experiences_path.read_bytes() + b'changed')
    elif failure == 'shape':
        case.directions['U'] = case.directions['U'][:, :, :-1]
        save_inputs(case, monkeypatch)
    elif failure == 'missing_model_config':
        (case.model_dir / 'config.json').unlink()
    elif failure == 'trainable_model':
        state.model.weight.requires_grad_(True)
    else:
        def acts(*args, **kwargs):
            value = state.acts(*args, **kwargs)
            if len(state.calls) == 7:
                if failure == 'missing_acts': return value[:-1]
                if failure == 'nan_acts': value[0, 0, 0] = float('nan')
                if failure == 'source_changed':
                    case.experiences_path.write_bytes(case.experiences_path.read_bytes() + b'\n')
            return value
        monkeypatch.setattr(module, 'last_prompt_acts', acts)
    with pytest.raises((ValueError, RuntimeError)):
        module.run(case.args)
    paths = list((case.root / 'runs').glob('**/run_meta.json'))
    assert len(paths) == 1
    out, meta = paths[0].parent, read_json(paths[0])
    assert meta['status'] == 'failed' and meta['end_time']
    assert not meta['judge_called'] and not meta['generation_performed']
    assert (out / 'artifact_hashes.json').exists() and (out / 'results.zip').exists()
    assert not (out / 'results.json').exists()
    if failure in ('hash', 'missing_model_config'):
        assert not state.model_loads and not state.calls
    if failure in ('missing_acts', 'nan_acts', 'source_changed'):
        assert len(state.calls) == 7
        neutral = torch.load(out / 'neutral_activations.pt', weights_only=True)
        assert len(neutral['per_memory']) == 5
        partial = torch.load(out / 'activation_differences.pt', weights_only=True)
        assert partial['mean_differences'].shape[0] >= 1
