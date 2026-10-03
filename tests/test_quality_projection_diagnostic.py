"""C18有限剥离入口：固定设计、同期评分、原张量与来源归档，全部CPU离线。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from antler.core.generate import generate
from test_core import D, model
from test_decode_strength import make_rows as historical_rows, phase_batches
from test_generate import Tokenizer
from test_phase_diagnostics import Tok, generation, make_vectors, prompt_value, traits


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e0_quality_projection')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_original_tensor_exact_and_projection_keeps_actual_norm(script, tmp_path):
    make_vectors(tmp_path / 'paired')
    base, _, _ = script.load_vectors(tmp_path / 'paired')
    q = torch.zeros(8, dtype=torch.float64)
    q[1], q[2], q[7] = -.5, -.5, 2 ** -.5
    snapshots = {dim: value.clone() for dim, value in base.items()}
    vectors, audit = script.build_injection_vectors(base, q)
    for dim in ('C', 'N'):
        expected = script.strength.arm_vector(base, dim, 'decode_mid')[21]
        original, projected = vectors[dim]['original'], vectors[dim]['projected']
        assert torch.equal(original, expected) and torch.equal(base[dim], snapshots[dim])
        assert script.tensor_sha256(original) == script.tensor_sha256(expected)
        assert projected.norm().item() == pytest.approx(original.norm().item(), rel=1e-6)
        assert abs(float(projected.double() @ q)) < 1e-6
        assert script.arm_vector(vectors, dim, 'original')[21] is original
        assert script.arm_vector(vectors, dim, 'projected')[21] is projected
        assert script.arm_vector(vectors, dim, 'zero') is None
        assert audit[dim]['same_l2_does_not_mean_same_trait_effect']
    assert [script.arm_phase(a) for a in script.ARMS] == ['all', 'decode', 'decode']
    assert script.COEFS['C']['original'] == script.COEFS['C']['projected'] == -.225
    assert script.COEFS['N']['original'] == script.COEFS['N']['projected'] == .200


def test_real_cpu_tiny_projected_decode_preserves_model_and_stage(script, model, monkeypatch):
    monkeypatch.setattr(script, 'LAYER', 2)
    monkeypatch.setattr(script.strength, 'LAYER', 2)
    monkeypatch.setattr(model.generation_config, 'eos_token_id', 1000)
    q = torch.zeros(D, dtype=torch.float64)
    q[0] = 1.
    vectors, _ = script.build_injection_vectors({'C': torch.ones(D) * .01, 'N': torch.ones(D) * .01}, q)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for arm in script.ARMS:
        trace = []
        result = generate(model, Tokenizer(), ['4 5 6'], steer=script.arm_vector(vectors, 'C', arm),
            steer_phase=script.arm_phase(arm), max_new_tokens=4, seed=1, batch_size=16,
            return_details=True, return_token_ids=True, phase_trace_out=trace)
        script.strength.validate_phase_traces(trace, result, script.arm_phase(arm), 4)
        assert result[0]['n_generated_tokens'] == 4
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, before[name])
               for name, p in model.named_parameters())
    assert all(not layer._forward_hooks for layer in model.model.layers) and not model._forward_pre_hooks


@pytest.fixture
def case(script, traits, tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    source, qdir, old_path = root / script.REGISTERED_VECTORS, root / script.REGISTERED_Q, root / script.REGISTERED_OLD_SAMPLES
    make_vectors(source)
    qdir.mkdir(parents=True)
    q = torch.zeros(8, dtype=torch.float64)
    q[1], q[2], q[7] = -.5, -.5, 2 ** -.5
    torch.save({'q': q}, qdir / 'statistics.pt')
    write_json(qdir / 'run_meta.json', {'status': 'completed', 'layer': 21})
    write_json(qdir / 'summary.json', {'direction_supported': True, 'bootstrap': {'complete': True},
                                     'targets': {p: {'candidate': True} for p in ('careless', 'nervous')}})
    write_json(qdir / 'folds.json', {'mock': True})
    torch.save({'mock': torch.zeros(1)}, qdir / 'response_mean_activations.pt')
    (qdir / 'replayed_inputs.jsonl').write_text('{}\n', encoding='utf-8')
    write_json(qdir / 'artifact_hashes.json', {p.name: script.sha256(p) for p in qdir.iterdir() if p.is_file()})
    old = historical_rows(script.strength, traits)
    # 非数字的旧评分故意保留，证明重复检查不读取它们。
    for row in old:
        row.update(trait='旧分禁止进入分析', coherence=None, task='ignore')
    old_path.parent.mkdir(parents=True)
    old_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in old), encoding='utf-8')
    for name in ('antler/mock.py', 'docs/RESEARCH_SPEC.md', 'docs/E0_QUALITY_PROJECTION_2026-10-04.md',
                 'mock-model/config.json', 'mock-model/tokenizer.json', 'config.yaml'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n', encoding='utf-8')
    upstream = [*source.joinpath('vectors').glob('*.pt'), *root.joinpath('mock-model').glob('*.json')]
    write_json(qdir / 'run_meta.json', {'status': 'completed', 'layer': 21,
        'data_hashes': {'/historical/antler/' + p.relative_to(root).as_posix(): script.sha256(p) for p in upstream}})
    write_json(qdir / 'artifact_hashes.json', {p.name: script.sha256(p) for p in qdir.iterdir()
                                             if p.is_file() and p.name != 'artifact_hashes.json'})
    kit = root / 'kit'
    for p, trait in traits.items():
        write_json(kit / 'trait_eval' / f'{p}.json', trait)
    monkeypatch.setattr(script, 'ROOT', root)
    monkeypatch.setattr(script, 'KIT', kit)
    monkeypatch.setattr(script, 'Q_HASHES', {name: script.sha256(qdir / name) for name in script.Q_HASHES})
    monkeypatch.setattr(script, 'REGISTERED_HASHES', {name: script.sha256(source / name) for name in script.REGISTERED_HASHES})
    monkeypatch.setattr(script, 'OLD_SAMPLES_SHA256', script.sha256(old_path))
    cfg = {'out_path': root / 'runs', 'model': 'mock-model', 'dtype': 'float32', 'device': 'cpu',
           'api_key': 'SECRET_MUST_NOT_LEAK'}
    args = SimpleNamespace(config=str(root / 'config.yaml'), vectors_dir=str(source),
                           q_source=str(qdir), old_samples=str(old_path))
    return SimpleNamespace(root=root, module=script, source=source, qdir=qdir, old=old,
                           old_path=old_path, cfg=cfg, args=args, traits=traits)


@pytest.mark.parametrize('change', ['q_hash', 'manifest_hash', 'candidate', 'incomplete', 'q_shape', 'path_escape', 'old_hash'])
def test_q_or_old_source_failures_stop_before_generation(case, monkeypatch, change):
    module, source = case.module, case.qdir
    if change == 'old_hash':
        case.old_path.write_bytes(case.old_path.read_bytes() + b'\n')
        with pytest.raises(ValueError): module.load_historical_texts(case.old_path, case.traits)
        return
    if change == 'q_hash': (source / 'statistics.pt').write_bytes(b'bad')
    elif change == 'manifest_hash': (source / 'folds.json').write_bytes(b'bad')
    elif change == 'path_escape':
        manifest = read_json(source / 'artifact_hashes.json')
        manifest['../escaped.json'] = '0' * 64
        write_json(source / 'artifact_hashes.json', manifest)
    else:
        if change in ('candidate', 'incomplete'):
            summary = read_json(source / 'summary.json')
            if change == 'candidate': summary['targets']['careless']['candidate'] = False
            else: summary['bootstrap']['complete'] = False
            write_json(source / 'summary.json', summary)
        else: torch.save({'q': torch.ones(3)}, source / 'statistics.pt')
        monkeypatch.setattr(module, 'Q_HASHES', {name: module.sha256(source / name) for name in module.Q_HASHES})
        write_json(source / 'artifact_hashes.json', {p.name: module.sha256(p) for p in source.iterdir()
                                                   if p.is_file() and p.name != 'artifact_hashes.json'})
    with pytest.raises(ValueError): module.load_quality_source(source, 8)


@pytest.mark.parametrize('kind', ['model', 'paired'])
def test_current_reused_upstream_must_match_q_source_binding(case, kind):
    path = case.root / 'mock-model' / 'config.json' if kind == 'model' else case.source / 'vectors' / 'dependable.pt'
    case.module.verify_quality_lineage(case.qdir, [path])
    # 形状无变化不足以确认来源；即使只是重新保存，也必须严格对应绑定字节。
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='来源绑定不一致'):
        case.module.verify_quality_lineage(case.qdir, [path])


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8), requires_grad=False)
        self.config = SimpleNamespace(num_hidden_layers=22, hidden_size=8,
            to_dict=lambda: {'num_hidden_layers': 22, 'hidden_size': 8})
        self.generation_config = SimpleNamespace(eos_token_id=[2, 3], to_dict=lambda: {'eos_token_id': [2, 3], 'use_cache': True})


@pytest.fixture
def mocked_run(case, monkeypatch):
    module = case.module
    state = SimpleNamespace(model=FrozenModel(), model_calls=0, generation_calls=[], score_calls=[], analysis_calls=0)
    monkeypatch.setattr(module, 'setup', lambda *a, **kw: (case.cfg, None, None, None))
    monkeypatch.setattr(module, 'load_trait', lambda p, split: deepcopy(case.traits[p]))
    monkeypatch.setattr(module, 'chat_prompt', lambda tok, q: q)
    monkeypatch.setattr(module.strength, 'chat_prompt', lambda tok, q: q)

    def load_model(*args):
        state.model_calls += 1
        return state.model, Tok()

    def fake_generate(model, tok, prompts, **kwargs):
        state.generation_calls.append((list(prompts), kwargs))
        kwargs['phase_trace_out'].extend(phase_batches(len(prompts), kwargs['steer_phase']))
        return [generation() for _ in prompts]

    def score_many(prompts):
        assert len(state.generation_calls) == 44
        assert sum(len(ps) for ps, _ in state.generation_calls) == 1088
        state.score_calls.append(list(prompts))
        return [prompt_value(p) for p in prompts]

    def judge(**kwargs):
        assert not Path(kwargs['cache_path']).exists()
        assert not Path(kwargs['response_audit_path']).exists()
        return SimpleNamespace(model='deepseek-flash', base_url='https://example.invalid',
                               max_tokens=16, extra_body={}, score_many=score_many)

    def analysis(rows, traits):
        state.analysis_calls += 1
        module.validate_rows(rows, traits)
        assert all(isinstance(row[m], (int, float)) for row in rows for m in ('trait', 'coherence', 'task'))
        return {'n_samples': len(rows), 'formal_e0_pass': False, 'capability_evaluated': False}

    monkeypatch.setattr(module, 'load_model', load_model)
    monkeypatch.setattr(module, 'generate', fake_generate)
    monkeypatch.setattr(module.strength, 'generate', fake_generate)
    monkeypatch.setattr(module, 'Judge', judge)
    monkeypatch.setattr(module, 'analyze', analysis)
    state.generate = fake_generate
    return state


def test_full_1080_three_arms_late_scoring_duplicate_report_and_zip(case, mocked_run):
    module, state = case.module, mocked_run
    out = module.run(case.args)
    assert state.model_calls == 1 and len(state.generation_calls) == 44 and len(state.score_calls) == 1
    assert state.analysis_calls == 1
    meta = read_json(out / 'run_meta.json')
    assert meta['status'] == 'completed' and meta['generated'] == 1080
    assert meta['old_scores_used'] is False and not meta['formal_e0_pass'] and not meta['capability_evaluated']
    assert len(read_json(out / 'q_lineage_checks.json')['verified_current_inputs']) == 6
    rows = [json.loads(line) for line in (out / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    module.validate_rows(rows, case.traits)
    assert set(r['arm'] for r in rows) == {'zero', 'original', 'projected'}
    assert all(r['question_index'] < 10 and r['sample_index'] < 3 and r['seed'] in (1, 2, 3) for r in rows)
    for _, kw in state.generation_calls:
        assert kw['batch_size'] == 16 and kw['temperature'] == kw['top_p'] == 1.
        assert kw['return_details'] and kw['return_token_ids']
    plan = read_json(out / 'score_plan.json')
    assert plan['n_inputs'] == 3240 and plan['seed'] == 0
    assert len(read_json(out / 'preflight.json')) == 8
    assert len(read_json(out / 'phase_traces.json')) == 36
    overlap = read_json(out / 'historical_text_overlap.json')
    assert overlap['n_compared'] == overlap['n_identical'] == 720 and overlap['old_scores_used'] is False
    assert len(overlap['details']) == 720
    assert not any('trait' in row or 'coherence' in row or 'task' in row for row in overlap['details'])
    injection = torch.load(out / 'injection_vectors.pt', weights_only=True)
    base, _, _ = module.load_vectors(case.source)
    for dim in ('C', 'N'):
        assert torch.equal(injection['vectors'][dim]['original'], module.strength.arm_vector(base, dim, 'decode_mid')[21])
        assert injection['vectors'][dim]['projected'].norm().item() == pytest.approx(
            injection['vectors'][dim]['original'].norm().item(), rel=1e-6)
    manifest = read_json(out / 'artifact_hashes.json')
    assert all(module.sha256(out / p) == digest for p, digest in manifest.items())
    with zipfile.ZipFile(out / 'results.zip') as archive:
        assert set(archive.namelist()) == set(manifest) | {'artifact_hashes.json'}
        assert all(b'SECRET_MUST_NOT_LEAK' not in archive.read(name) for name in archive.namelist())


def test_full_runner_with_real_statistics_accepts_twenty_question_sources(case, mocked_run, monkeypatch):
    from antler.traits.quality_projection import analyze
    monkeypatch.setattr(case.module, 'analyze', analyze)
    out = case.module.run(case.args)
    report = read_json(out / 'assessment.json')
    assert read_json(out / 'run_meta.json')['status'] == 'completed'
    assert report['n_samples'] == 1080 and len(report['conditions']) == 8
    assert len(report['candidates']) == 4 and len(report['method_decisions']) == 2
    assert report['quality_comparisons']['n_tests'] == 8
    assert report['formal_e0_pass'] is False and report['capability_evaluated'] is False
    # 模拟三臂文本完全相同，真实统计不应凭空制造人格增益。
    assert all(not candidate['pooled']['numeric_criteria_met'] for candidate in report['candidates'])


@pytest.mark.parametrize('failure', ['source_drift', 'q_drift', 'trainable', 'preflight', 'phase', 'score_missing', 'score_exception'])
def test_failure_stops_preserves_raw_inputs_no_retry_or_early_score(case, mocked_run, monkeypatch, failure):
    module, state = case.module, mocked_run
    if failure == 'trainable': state.model.weight.requires_grad_(True)
    elif failure.startswith('score_'):
        original = module.Judge
        def judge(**kwargs):
            value = original(**kwargs)
            base_score = value.score_many
            def score(prompts):
                values = base_score(prompts)
                if failure == 'score_exception': raise RuntimeError('SECRET_MUST_NOT_LEAK')
                values[0] = None
                return values
            value.score_many = score
            return value
        monkeypatch.setattr(module, 'Judge', judge)
    else:
        def generation_failure(*args, **kwargs):
            result = state.generate(*args, **kwargs)
            short = kwargs['max_new_tokens'] == 16
            if failure == 'preflight' and short and kwargs['steer_phase'] == 'decode':
                result[0] = generation([8, 2])
            if not short:
                if failure == 'source_drift': Path(case.args.config).write_text('changed\n', encoding='utf-8')
                elif failure == 'q_drift': (case.qdir / 'statistics.pt').write_bytes(b'changed')
                elif failure == 'phase' and kwargs['steer_phase'] == 'decode':
                    kwargs['phase_trace_out'][0]['trace'][0]['applied'] = True
            return result
        monkeypatch.setattr(module, 'generate', generation_failure)
        monkeypatch.setattr(module.strength, 'generate', generation_failure)
    with pytest.raises((ValueError, RuntimeError)):
        module.run(case.args)
    out = next((case.cfg['out_path'] / 'quality_projection').iterdir())
    meta_text = (out / 'run_meta.json').read_text(encoding='utf-8')
    meta = json.loads(meta_text)
    assert meta['status'] == 'failed' and meta['end_time'] and state.analysis_calls == 0
    assert 'SECRET_MUST_NOT_LEAK' not in meta_text and (out / 'results.zip').exists()
    assert not (out / 'assessment.json').exists()
    if failure == 'trainable': assert not state.generation_calls and not state.score_calls
    elif failure == 'preflight': assert len(state.generation_calls) == 4 and not state.score_calls
    elif failure in ('source_drift', 'q_drift'):
        assert len(state.generation_calls) == 9 and not state.score_calls
        assert meta['generated'] == 30
    elif failure == 'phase':
        assert len(state.generation_calls) == 10 and not state.score_calls and meta['generated'] == 30
    else: assert len(state.generation_calls) == 44 and len(state.score_calls) == 1 and meta['generated'] == 1080
