"""C18重放入口的CPU离线测试：来源、token切片、冻结、失败留档。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch
from transformers import BatchEncoding

from antler.core.activations import response_mean_acts
from antler.core.hooks import Recorder
from antler.core.model import get_layers
from antler.traits.data import DIMS
from antler.traits.vectors import extraction_rows
from test_core import D, L, model


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e0_quality_direction')


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _save_sources(case, monkeypatch, *, repin=True):
    for pole in case.module.POLES:
        path = case.source / 'vectors' / f'{pole}_samples.json'
        _json(path, case.samples[pole])
        if repin:
            for record in case.records:
                if record['pole'] == pole:
                    record['source_sha256'] = case.module.sha256(path)
    case.c17.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in case.records),
                         encoding='utf-8', newline='\n')
    monkeypatch.setattr(case.module, 'C17_SHA256', case.module.sha256(case.c17))


@pytest.fixture
def case(tmp_path, diagnostic, monkeypatch):
    source, vectors = tmp_path / 'source', tmp_path / 'paired'
    (source / 'vectors').mkdir(parents=True)
    vectors.mkdir()
    monkeypatch.setattr(diagnostic, 'LAYER', 2)
    u = torch.eye(5, D).repeat(L, 1, 1)
    payload = {'U': u, 's': torch.full((L,), 2.), 'dims': list(DIMS)}
    torch.save(payload, source / 'directions.pt')
    monkeypatch.setattr(diagnostic, 'DIRECTIONS_SHA256', diagnostic.sha256(source / 'directions.pt'))
    vector_hashes, samples, records = {}, {}, []
    for pi, pole in enumerate(diagnostic.POLES):
        vector = torch.arange(1, D + 1).float().repeat(L, 1) * (pi + 1)
        torch.save(vector, vectors / f'{pole}.pt')
        vector_hashes[pole] = diagnostic.sha256(vectors / f'{pole}.pt')
        samples[pole] = []
        for row in extraction_rows(pole):
            score = 70. + row['rollout'] * 5 if row['side'] == 'pos' else 20. + row['rollout'] * 2
            sample = {'side': row['side'], 'question': row['question'],
                      'answer': f'模拟回答 {pole} {row["source_index"]}', 'score': score}
            samples[pole].append(sample)
            records.append({'pole': pole, 'source_index': row['source_index'],
                            'source_sha256': '', 'side': row['side'], 'question': row['question'],
                            'answer': sample['answer'], 'trait_score': score, 'coherence': 75. + row['rollout'] * 10})
    monkeypatch.setattr(diagnostic, 'VECTOR_SHA256', vector_hashes)
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    _json(model_dir / 'config.json', {'hidden_size': D, 'num_hidden_layers': L})
    _json(model_dir / 'tokenizer.json', {'synthetic': True})
    config = tmp_path / 'config.yaml'
    config.write_text('model: synthetic\n', encoding='utf-8')
    result = SimpleNamespace(root=tmp_path, source=source, vectors=vectors, module=diagnostic,
                             samples=samples, records=records, c17=source / 'extract_coherence_samples.jsonl',
                             directions=payload, config=config, model_dir=model_dir)
    result.cfg = {'model': str(model_dir), 'out_path': tmp_path / 'runs', 'device': 'cpu',
                  'dtype': 'float32', 'n_rollouts': 2, 'api_key': 'SECRET_EXAMPLE'}
    result.args = SimpleNamespace(config=str(config), source_out=str(source), pole_vectors=str(vectors))
    _save_sources(result, monkeypatch)
    return result


def test_prepare_complete_pool_identity_and_fixed_folds(case):
    result = case.module.prepare_inputs(case.cfg, case.source, case.vectors)
    assert len(result['rows']) == 1600
    assert result['s_layer'] == 2 and result['hidden_size'] == D and result['n_layers'] == L
    assert set(result['reference_vectors']) == set(case.module.POLES)
    for pole in case.module.POLES:
        rows = [r for r in result['rows'] if r['pole'] == pole]
        assert [r['source_index'] for r in rows] == list(range(400))
        assert len({r['question'] for r in rows}) == 20
        assert {r['fold'] for r in rows} == {'A', 'B'}
        assert [r['system'] for r in rows] == [r['system'] for r in extraction_rows(pole)]
        assert [r['answer'] for r in rows] == [r['answer'] for r in case.samples[pole]]
    assert all(case.module.sha256(path) == digest for path, digest in result['hashes'].items())
    assert not any('trait_eval' in path for path in result['hashes'])


@pytest.mark.parametrize('change', ['source_sha256', 'side', 'question', 'answer', 'trait_score',
                                  'missing_coherence', 'nan_coherence', 'missing_trait', 'bool_trait',
                                  'duplicate', 'missing', 'extra_pole', 'bool_index'])
def test_c17_mismatch_missing_or_invalid_scores_stop(case, monkeypatch, change):
    row = case.records[0]
    if change == 'source_sha256': row['source_sha256'] = '0' * 64
    elif change in ('side', 'question', 'answer'): row[change] += ' changed'
    elif change == 'trait_score': row['trait_score'] += 1
    elif change == 'missing_coherence': row.pop('coherence')
    elif change == 'nan_coherence': row['coherence'] = float('nan')
    elif change == 'missing_trait': row['trait_score'] = None
    elif change == 'bool_trait': row['trait_score'] = True
    elif change == 'duplicate': case.records[1] = deepcopy(row)
    elif change == 'missing': case.records.pop()
    elif change == 'extra_pole': row['pole'] = 'inventive'
    else: row['source_index'] = True
    _save_sources(case, monkeypatch, repin=False)
    with pytest.raises(ValueError):
        case.module.prepare_inputs(case.cfg, case.source, case.vectors)


@pytest.mark.parametrize('kind', ['c17', 'directions', 'vector', 'source'])
def test_fixed_bytes_reject_modified_file(case, kind):
    path = {'c17': case.c17, 'directions': case.source / 'directions.pt',
            'vector': case.vectors / 'careless.pt',
            'source': case.source / 'vectors' / 'careless_samples.json'}[kind]
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        case.module.prepare_inputs(case.cfg, case.source, case.vectors)


@pytest.mark.parametrize('change', ['shape', 'dims', 'scale', 'dtype', 'not_unit', 'model', 'layer'])
def test_direction_contract_before_model(case, monkeypatch, change):
    value = case.directions
    if change == 'shape': value['U'] = value['U'][0]
    elif change == 'dims': value['dims'].reverse()
    elif change == 'scale': value['s'][2] = 0
    elif change == 'dtype': value['s'] = value['s'].long()
    elif change == 'not_unit': value['U'][0, 0] *= 2
    elif change == 'model': value['meta'] = {'model': 'different'}
    else: value['U'], value['s'] = value['U'][:2], value['s'][:2]
    path = case.source / 'directions.pt'
    torch.save(value, path)
    monkeypatch.setattr(case.module, 'DIRECTIONS_SHA256', case.module.sha256(path))
    with pytest.raises(ValueError):
        case.module.prepare_inputs(case.cfg, case.source, case.vectors)


class CharacterTokenizer:
    """字符分词器支持core真实采用的字符串和批处理两条路径。"""
    padding_side = 'left'

    def encode(self, text):
        return [3 + ord(c) % 60 for c in text]

    def __call__(self, text, **kwargs):
        assert kwargs.get('add_special_tokens') is False
        if isinstance(text, str):
            return {'input_ids': self.encode(text)}
        ids = [self.encode(t) for t in text]
        width = max(map(len, ids))
        assert self.padding_side == 'right'
        return BatchEncoding({'input_ids': torch.tensor([x + [0] * (width - len(x)) for x in ids]),
                              'attention_mask': torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])})

    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages) + '\n助手：'


def test_response_token_span_uses_concat_minus_prompt_boundary(diagnostic):
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text):
            return super().encode(text.replace('ab', 'Z'))
    tok = MergingTokenizer()
    spans = diagnostic.replay_token_spans(tok, ['a', 'long'], ['bC', 'D'])
    assert spans[0]['replayed_prompt_tokens'] == 1
    assert spans[0]['replayed_total_tokens'] == 2
    assert spans[0]['replayed_response_tokens'] == 1
    assert len(tok('bC', add_special_tokens=False)['input_ids']) == 2
    assert spans[1]['replayed_response_tokens'] == 1
    assert tok.padding_side == 'left'
    with pytest.raises(ValueError, match='为空'):
        diagnostic.replay_token_spans(tok, ['a'], ['b'])
    assert tok.padding_side == 'left'


def test_real_tiny_model_replay_matches_exact_recorded_slice(diagnostic, model):
    tok = CharacterTokenizer()
    prompts, answers = ['abc', 'd'], ['EF', 'GHIJ']
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    spans = diagnostic.replay_token_spans(tok, prompts, answers)
    acts = response_mean_acts(model, tok, prompts, answers, layers=[2], batch_size=16)
    checked = diagnostic.validate_activations(acts, 2, D)
    tok.padding_side = 'right'
    enc = tok([p + a for p, a in zip(prompts, answers)], return_tensors='pt', padding=True, add_special_tokens=False)
    with torch.no_grad(), Recorder(model, [2]) as recorder:
        model(**enc)
    for i, span in enumerate(spans):
        start, end = span['replayed_prompt_tokens'], span['replayed_total_tokens']
        assert end - start == span['replayed_response_tokens']
        assert torch.equal(checked[i], recorder.acts[2][i, start:end].float().mean(0))
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, before[name])
               for name, p in model.named_parameters())
    assert all(not layer._forward_hooks for layer in get_layers(model))


@pytest.mark.parametrize('kind', ['wrong_shape', 'integer', 'nan', 'overflow'])
def test_activation_failures_stop(diagnostic, kind):
    value = torch.ones(16, 1, D)
    if kind == 'wrong_shape': value = value[:, 0]
    elif kind == 'integer': value = value.long()
    elif kind == 'nan': value[0, 0, 0] = float('nan')
    else: value = torch.full((16, 1, D), 1e300, dtype=torch.float64)
    with pytest.raises(ValueError):
        diagnostic.validate_activations(value, 16, D)


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(D), requires_grad=False)
        self.config = SimpleNamespace(num_hidden_layers=L, hidden_size=D,
                                      to_dict=lambda: {'hidden_size': D, 'num_hidden_layers': L})


@pytest.fixture
def mocked_run(case, monkeypatch):
    module = case.module
    fake_root = case.root / 'repo'
    for relative in ('antler/core.py', 'experiments/_common.py', 'experiments/e0_diagnostics.py',
                     'experiments/e0_extract_coherence.py', 'experiments/e0_refilter.py',
                     'docs/RESEARCH_SPEC.md', 'docs/E0_QUALITY_DIRECTION_2026-10-04.md'):
        path = fake_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('固定登记\n', encoding='utf-8')
    monkeypatch.setattr(module, 'ROOT', fake_root)
    state = SimpleNamespace(model=FrozenModel(), model_loads=0, calls=[], stats_calls=0)

    def setup(args, *, need_model, need_judge):
        assert not need_model and not need_judge
        return case.cfg, None, None, None

    def load_model(*args):
        state.model_loads += 1
        fold_files = list((case.root / 'runs').rglob('folds.json'))
        assert len(fold_files) == 1 and len(_read(fold_files[0])['rows']) == 1600
        return state.model, CharacterTokenizer()

    def acts(model, tok, prompts, answers, *, layers, batch_size):
        assert model is state.model and layers == [2] and batch_size == 16
        assert len(prompts) == len(answers) == 16
        state.calls.append((list(prompts), list(answers)))
        return torch.full((16, 1, D), float(len(state.calls)))

    def stats(rows, activations, tokens, scale, references, *, bootstrap_samples, seed):
        state.stats_calls += 1
        assert len(rows) == len(tokens) == 1600 and activations.shape == (1600, D)
        assert all(v > 0 for v in tokens) and scale == 2
        assert bootstrap_samples == 1000 and seed == 0
        assert set(references) == set(module.POLES)
        return {'summary': {'scope': '模拟统计', 'formal_e0_pass': False, 'bootstrap': {'complete': True}},
                'artifacts': {'q': torch.ones(D)}}

    monkeypatch.setattr(module, 'setup', setup)
    monkeypatch.setattr(module, 'load_model', load_model)
    monkeypatch.setattr(module, 'response_mean_acts', acts)
    monkeypatch.setattr(module, 'analyze_quality_direction', stats)
    state.acts = acts
    return state


def test_run_fixed_1600_no_generation_no_judge_and_manifest(case, mocked_run):
    out = case.module.run(case.args)
    state = mocked_run
    assert state.model_loads == 1 and len(state.calls) == 100 and state.stats_calls == 1
    meta = _read(out / 'run_meta.json')
    assert meta['status'] == 'completed' and meta['answers_completed'] == 1600
    assert meta['batches_completed'] == 100 and meta['end_time']
    assert meta['judge_called'] is False and meta['generation_performed'] is False
    assert meta['injection_performed'] is False and meta['trait_eval_used'] is False
    assert meta['formal_e0_pass'] is False and meta['historical_token_ids_available'] is False
    assert 'SECRET_EXAMPLE' not in (out / 'run_meta.json').read_text(encoding='utf-8')
    rows = [json.loads(line) for line in (out / 'replayed_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 1600 and [r['answer'] for r in rows] == [a for _, answers in state.calls for a in answers]
    assert [r['rendered_prompt'] for r in rows] == [p for prompts, _ in state.calls for p in prompts]
    assert all(r['replayed_response_tokens'] == len(r['answer']) for r in rows)
    acts = torch.load(out / 'response_mean_activations.pt', weights_only=True)
    assert acts['response_mean_activations'].shape == (1600, D)
    assert len(acts['rows']) == 1600
    assert acts['folds_sha256'] == case.module.sha256(out / 'folds.json')
    manifest = _read(out / 'artifact_hashes.json')
    assert all(case.module.sha256(out / name) == digest for name, digest in manifest.items())
    with zipfile.ZipFile(out / 'results.zip') as archive:
        assert set(archive.namelist()) == set(manifest) | {'artifact_hashes.json'}
        assert archive.read('run_meta.json') == (out / 'run_meta.json').read_bytes()
    assert not any('trait_eval' in path for path in meta['data_hashes'])


@pytest.mark.parametrize('failure', ['source_hash', 'trainable', 'model_shape', 'nonfinite', 'source_drift', 'stats'])
def test_failure_retains_artifacts_and_never_retries(case, mocked_run, monkeypatch, failure):
    state, module = mocked_run, case.module
    if failure == 'source_hash': case.c17.write_bytes(case.c17.read_bytes() + b'\n')
    elif failure == 'trainable': state.model.weight.requires_grad_(True)
    elif failure == 'model_shape': state.model.config.hidden_size += 1
    elif failure == 'stats':
        def fail(*args, **kwargs):
            state.stats_calls += 1
            raise ValueError('模拟重建不一致')
        monkeypatch.setattr(module, 'analyze_quality_direction', fail)
    else:
        def acts(*args, **kwargs):
            value = state.acts(*args, **kwargs)
            if len(state.calls) == 2:
                if failure == 'nonfinite': value[0, 0, 0] = float('nan')
                else: case.config.write_text('changed\n', encoding='utf-8')
            return value
        monkeypatch.setattr(module, 'response_mean_acts', acts)
    with pytest.raises(ValueError):
        module.run(case.args)
    paths = list((case.root / 'runs').rglob('run_meta.json'))
    assert len(paths) == 1
    meta, out = _read(paths[0]), paths[0].parent
    assert meta['status'] == 'failed' and meta['end_time']
    assert (out / 'results.zip').exists() and (out / 'artifact_hashes.json').exists()
    assert not (out / 'summary.json').exists()
    if failure == 'source_hash': assert state.model_loads == 0
    if failure in ('trainable', 'model_shape'): assert len(state.calls) == 0
    if failure in ('nonfinite', 'source_drift'):
        assert len(state.calls) == 2 and state.stats_calls == 0
        partial = torch.load(out / 'response_mean_activations.pt', weights_only=True)
        assert partial['response_mean_activations'].shape[0] in (16, 32)
    if failure == 'stats':
        assert state.stats_calls == 1 and len(state.calls) == 100
        diagnostic = _read(out / 'statistics_failure.json')
        assert diagnostic['error_type'] == 'ValueError' and diagnostic['error'] == '模拟重建不一致'


def test_bootstrap_failure_keeps_diagnostic_result_but_marks_run_failed(case, mocked_run, monkeypatch):
    def incomplete(*args, **kwargs):
        mocked_run.stats_calls += 1
        return {'summary': {'bootstrap': {'complete': False, 'rho_ci95': None,
                                         'failures': [{'replicate': 7, 'error': '秩不足'}]},
                            'targets': {'careless': {'candidate': False}, 'nervous': {'candidate': False}}},
                'artifacts': {'q': torch.ones(D)}}
    monkeypatch.setattr(case.module, 'analyze_quality_direction', incomplete)
    with pytest.raises(ValueError, match='诊断不完整'):
        case.module.run(case.args)
    paths = list((case.root / 'runs').rglob('run_meta.json'))
    assert len(paths) == 1 and mocked_run.stats_calls == 1
    meta, out = _read(paths[0]), paths[0].parent
    assert meta['status'] == 'failed' and meta['bootstrap_complete'] is False
    assert meta['answers_completed'] == 1600 and meta['stage'] == 'cpu_statistics'
    assert _read(out / 'summary.json')['bootstrap']['failures'][0]['replicate'] == 7
    assert (out / 'statistics.pt').exists() and (out / 'results.zip').exists()
