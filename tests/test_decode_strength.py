"""C19固定强度诊断：离线身份、阶段、统计边界与失败留档测试。"""
from collections import Counter
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from test_core import model, D
from test_generate import Tokenizer
from test_phase_diagnostics import Tok, generation, make_vectors, prompt_value, traits


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e0_decode_strength')


def make_rows(script, traits):
    rows = []
    for dim in script.COEFS:
        for pole, target in zip(('high', 'low'), script.BIG_FIVE[dim]):
            for arm in script.ARMS:
                for seed in script.SEEDS:
                    for sample in range(3):
                        for q in range(10):
                            rows.append({'dim': dim, 'pole': pole, 'target': target, 'arm': arm,
                                         'question_index': q, 'sample_index': sample, 'seed': seed,
                                         'q': traits[target]['questions'][q], 'a': '7', 'layer': 21,
                                         'coef': script.COEFS[dim][arm], 'steer_phase': script.arm_phase(arm),
                                         'axis': 'pole', 'split': 'calib', 'token_ids': [7, 2],
                                         'n_generated_tokens': 2, 'finish_reason': 'eos',
                                         'trait': 0 if arm == 'zero' else 25, 'coherence': 90, 'task': 90})
    return rows


def phase_batches(n, phase, n_steps=2):
    trace = [{'phase': 'prefill' if i == 0 else 'decode', 'cached_tokens': 0 if i == 0 else 4+i,
              'cache_position_start': 0 if i == 0 else 4+i,
              'applied': (i == 0) == (phase == 'prefill')} for i in range(n_steps)]
    return [{'batch_start': i, 'trace': [] if phase == 'all' else copy.deepcopy(trace)} for i in range(0, n, 16)]


def test_fixed_design_and_arm_vectors(script, traits, tmp_path):
    rows = make_rows(script, traits)
    script.validate_rows(rows, traits)
    assert len(rows) == 1800
    assert Counter((r['dim'], r['pole'], r['seed'], r['arm']) for r in rows) == {
        (d, p, s, a): 30 for d in ('C', 'N') for p in ('high', 'low') for s in (1, 2, 3) for a in script.ARMS}
    make_vectors(tmp_path / 'paired')
    vectors, shape, files = script.load_vectors(tmp_path / 'paired')
    assert shape == (22, 5, 8) and len(files) == 5
    for dim in ('C', 'N'):
        assert script.arm_vector(vectors, dim, 'zero') is None
        assert script.arm_vector(vectors, dim, 'all')[21] is vectors[dim]
        assert script.arm_vector(vectors, dim, 'decode_base')[21] is vectors[dim]
        for arm in script.DECODE_ARMS:
            value = script.arm_vector(vectors, dim, arm)[21]
            assert value.norm().item() == pytest.approx(abs(script.COEFS[dim][arm]) * 3)
            assert torch.allclose(value / value.norm(), vectors[dim] / vectors[dim].norm())
            assert script.arm_phase(arm) == 'decode'
    assert script.COEFS['C']['decode_upper'] == -.300
    assert script.COEFS['N']['decode_mid'] == .200


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'seed0', 'seed_bool', 'question_bool',
    'question_float', 'test_index', 'sample3', 'arm', 'question', 'target', 'coef_bool', 'coef',
    'layer_bool', 'layer', 'split', 'axis', 'steer_phase'])
def test_identity_protocol_failures(script, traits, change):
    rows = make_rows(script, traits)
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    else:
        key, value = {
            'seed0': ('seed', 0), 'seed_bool': ('seed', True), 'question_bool': ('question_index', False),
            'question_float': ('question_index', 0.), 'test_index': ('question_index', 10),
            'sample3': ('sample_index', 3), 'arm': ('arm', 'prefill'), 'question': ('q', '假题'),
            'target': ('target', 'careless'), 'coef_bool': ('coef', False), 'coef': ('coef', -.175),
            'layer_bool': ('layer', True), 'layer': ('layer', 20), 'split': ('split', 'test'),
            'axis': ('axis', 'bipolar'), 'steer_phase': ('steer_phase', 'decode'),
        }[change]
        rows[0][key] = value
    with pytest.raises(ValueError):
        script.validate_rows(rows, traits)


@pytest.mark.parametrize('phase', ['all', 'prefill', 'decode'])
def test_phase_trace_accepts_batched_and_deferred_eos(script, phase):
    items = [generation() for _ in range(30)]
    script.validate_phase_traces(phase_batches(30, phase), items, phase, 320)
    script.validate_phase_traces(phase_batches(30, phase, 5), items, phase, 320)
    batches = phase_batches(30, phase)
    for batch in batches:
        for entry in batch['trace']:
            entry['cache_position_start'] = None
    script.validate_phase_traces(batches, items, phase, 320)


@pytest.mark.parametrize('change', ['missing_batch', 'bad_start', 'bool_start', 'short', 'long',
    'field', 'first_decode', 'empty_decode', 'bool_cache', 'position', 'applied', 'bool_applied',
    'nonadvancing', 'all_nonempty'])
def test_bad_phase_trace_stops(script, change):
    phase = 'decode'
    batches = phase_batches(30, phase, 3)
    if change == 'missing_batch': batches.pop()
    elif change == 'bad_start': batches[0]['batch_start'] = 16
    elif change == 'bool_start': batches[0]['batch_start'] = False
    elif change == 'short': batches[0]['trace'] = batches[0]['trace'][:1]
    elif change == 'long': batches = phase_batches(30, phase, 321)
    elif change == 'field': batches[0]['trace'][0].pop('cached_tokens')
    elif change == 'first_decode': batches[0]['trace'][0]['phase'] = 'decode'
    elif change == 'empty_decode': batches[0]['trace'][1]['cached_tokens'] = 0
    elif change == 'bool_cache': batches[0]['trace'][0]['cached_tokens'] = False
    elif change == 'position': batches[0]['trace'][1]['cache_position_start'] = 1
    elif change == 'applied': batches[0]['trace'][0]['applied'] = True
    elif change == 'bool_applied': batches[0]['trace'][0]['applied'] = 0
    elif change == 'nonadvancing':
        batches[0]['trace'][2]['cached_tokens'] = batches[0]['trace'][2]['cache_position_start'] = 5
    elif change == 'all_nonempty': phase = 'all'
    with pytest.raises(ValueError):
        script.validate_phase_traces(batches, [generation() for _ in range(30)], phase, 320)


def test_real_random_tiny_cpu_preflight_eight_generations(script, model, traits, monkeypatch, tmp_path):
    class ChatTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return '4 5 6'
    monkeypatch.setattr(script, 'LAYER', 2)
    # 关闭随机EOS，仍使用合法但词表外的停止ID，以稳定验证16-token上限。
    monkeypatch.setattr(model.generation_config, 'eos_token_id', 1000)
    vectors = {'C': torch.ones(D), 'N': torch.ones(D)}
    records = script.preflight(model, ChatTokenizer(), vectors, traits, {1000}, tmp_path)
    assert len(records) == 8
    for dim in ('C', 'N'):
        own = [r for r in records if r['dim'] == dim]
        assert all(r['generations'] == own[0]['generations'] for r in own)
        assert all(r['generations'][0]['n_generated_tokens'] == 16 for r in own)
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert not model._forward_pre_hooks


def test_exact_sign_flip_is_two_sided_symmetric_and_handles_zero(script):
    assert script.exact_sign_flip([1] * 10) == 2 / 1024
    assert script.exact_sign_flip([-1] * 10) == 2 / 1024
    assert script.exact_sign_flip([0] * 10) == 1
    mixed = [-2.3, 0, .1, 2, 3, 1, 5, 1, 2, -1]
    assert script.exact_sign_flip(mixed) == script.exact_sign_flip([-v for v in mixed])
    with pytest.raises(ValueError): script.exact_sign_flip([1] * 9)


def test_analysis_uses_ten_questions_seed_means_and_fixed_quality_family(script, traits):
    rows = make_rows(script, traits)
    for r in rows:
        q, seed, arm = r['question_index'], r['seed'], r['arm']
        if arm in script.DECODE_ARMS:
            r['trait'] = 24 + q + seed - 2 + r['sample_index']
            r['coherence'], r['task'] = 95, 94
    report = script.analyze(rows, traits)
    assert len(report['conditions']) == 16 and len(report['candidates']) == 6
    assert len(report['sensitivity']) == 18
    assert report['formal_e0_pass'] is False and report['seed_stddev_ddof'] == 1
    for condition in report['conditions']:
        if condition['arm'] == 'all': continue
        assert condition['pooled']['versus_zero']['trait']['mean'] == 29.5
        assert condition['pooled']['versus_all']['trait']['mean'] == 4.5
        assert condition['pooled']['versus_zero']['trait']['n_questions'] == 10
        assert condition['pooled']['versus_zero']['trait']['n_boot'] == 10000
        assert [r['versus_zero']['trait']['mean'] for r in condition['per_seed']] == [28.5, 29.5, 30.5]
        assert condition['seed_stddev']['versus_zero']['trait'] == 1
        assert condition['seed_stddev']['versus_all']['trait'] == 1
    family = report['quality_comparisons']
    assert family['n_tests'] == len(family['tests']) == 24
    assert len({(t['dim'], t['arm'], t['pole'], t['metric']) for t in family['tests']}) == 24
    assert all(t['p'] == t['adjusted_p'] == 2/1024 and t['reject'] for t in family['tests'])
    assert all(t['quality_improvement_supported'] for t in family['tests'])
    # 均值4.5接近，但人格差CI越过5，不能声称相近效应下质量改善。
    assert all(c['target_trait_versus_all']['mean_close_descriptive'] for c in report['candidates'])
    assert not any(t['similar_trait_quality_improvement_supported'] for t in family['tests'])
    assert report['selected']['C']['arm'] == report['selected']['N']['arm'] == 'decode_base'


@pytest.mark.parametrize('scenario', ['both_poles', 'seed_failure', 'sensitivity', 'none', 'close_supported'])
def test_candidate_boundaries_sensitivity_and_closeness(script, traits, scenario):
    rows = make_rows(script, traits)
    for r in rows:
        if r['arm'] not in script.DECODE_ARMS: continue
        if scenario == 'both_poles' and r['arm'] == 'decode_base' and r['pole'] == 'high' and r['dim'] == 'C':
            r['task'] = 79
        elif scenario == 'seed_failure':
            r['trait'] = {1: 10, 2: 25, 3: 40}[r['seed']]
        elif scenario == 'sensitivity':
            r['trait'] = {'decode_base': 15, 'decode_mid': 20, 'decode_upper': 10}[r['arm']]
            r['coherence'], r['task'] = 80, 80
        elif scenario == 'none': r['coherence'] = 74
        elif scenario == 'close_supported':
            r['trait'] = 30
            r['coherence'] = r['task'] = 95
    report = script.analyze(rows, traits)
    if scenario == 'both_poles':
        c = next(c for c in report['candidates'] if c['dim'] == 'C' and c['arm'] == 'decode_base')
        assert 'high_task_drop_le10' in c['pooled']['failed_checks']
        assert report['selected']['C']['arm'] == 'decode_mid'
    elif scenario == 'seed_failure':
        assert all(c['pooled']['numeric_criteria_met'] for c in report['candidates'])
        for candidate in report['candidates']:
            assert candidate['per_seed'][0]['failed_checks'] == ['target_trait_gain', 'target_trait_ci_low_gt10']
        assert report['selected']['C']['arm'] == 'decode_base'
    elif scenario == 'sensitivity':
        table = {(r['dim'], r['arm'], r['trait_mean_cutoff']): r for r in report['sensitivity']}
        assert not table['C', 'decode_base', 20]['numeric_criteria_met']
        assert table['C', 'decode_base', 15]['numeric_criteria_met']
        assert table['C', 'decode_base', 10]['numeric_criteria_met']
        assert table['C', 'decode_mid', 20]['numeric_criteria_met']
        assert not table['C', 'decode_upper', 10]['numeric_criteria_met']
        assert table['C', 'decode_upper', 10]['failed_checks'] == ['target_trait_ci_low_gt10']
        assert table['C', 'decode_mid', 20]['quality_boundary_uncertainty']
        assert report['selected']['C']['arm'] == 'decode_mid'
    elif scenario == 'none': assert report['selected'] == {'C': None, 'N': None}
    elif scenario == 'close_supported':
        assert all(t['similar_trait_quality_improvement_supported'] for t in report['quality_comparisons']['tests'])
        assert all(c['target_trait_versus_all']['ci_within_minus5_plus5'] for c in report['candidates'])
        assert all('equivalent' not in c['target_trait_versus_all'] for c in report['candidates'])


def test_quality_ci_boundary_is_uncertainty_not_an_extra_gate(script, traits):
    rows = make_rows(script, traits)
    for r in rows:
        if r['arm'] in script.DECODE_ARMS:
            r['task'] = 50 if r['question_index'] == 0 else 90
    report = script.analyze(rows, traits)
    assert all(c['pooled']['numeric_criteria_met'] for c in report['candidates'])
    assert all(c['pooled']['quality_boundary_uncertainty'] for c in report['candidates'])


def test_positive_quality_ci_requires_bh_support_for_similar_effect_claim(script, traits, monkeypatch):
    rows = make_rows(script, traits)
    for r in rows:
        if r['arm'] in script.DECODE_ARMS:
            r['coherence'] = r['task'] = 95
    def no_rejections(p_values, *, q):
        assert len(p_values) == 24 and q == .05
        return {'adjusted_p': [.1] * 24, 'reject': [False] * 24, 'q': q}
    monkeypatch.setattr(script, 'benjamini_hochberg', no_rejections)
    report = script.analyze(rows, traits)
    assert all(c['target_trait_versus_all']['ci_within_minus5_plus5'] for c in report['candidates'])
    assert all(t['ci_low'] > 0 for t in report['quality_comparisons']['tests'])
    assert not any(t['similar_trait_quality_improvement_supported'] for t in report['quality_comparisons']['tests'])


@pytest.mark.parametrize('value', [None, True, float('nan'), float('inf'), -1, 101, '80'])
def test_analysis_rejects_missing_or_invalid_scores(script, traits, value):
    rows = make_rows(script, traits)
    rows[0]['trait'] = value
    with pytest.raises(ValueError): script.analyze(rows, traits)


@pytest.mark.parametrize('failure', [None, 'source_path', 'source_hash', 'local_model', 'preflight', 'preflight_trace',
    'phase_trace', 'token_text', 'missing_score', 'secret_error', 'source_changed', 'source_before_score', 'judge_changed'])
def test_run_full_matrix_or_preserved_failure_without_retry(script, traits, monkeypatch, tmp_path, failure):
    root = tmp_path / 'repo'
    source = root / script.REGISTERED_VECTORS
    make_vectors(source)
    (root / 'antler').mkdir()
    (root / 'antler' / 'synthetic.py').write_text('# 合成来源\n', encoding='utf-8')
    (root / 'docs').mkdir()
    for name in ('RESEARCH_SPEC.md', 'E0_DECODE_STRENGTH_2026-10-04.md'):
        (root / 'docs' / name).write_text('合成登记\n', encoding='utf-8')
    kit = root / 'kit'
    (kit / 'trait_eval').mkdir(parents=True)
    for name, value in traits.items():
        (kit / 'trait_eval' / f'{name}.json').write_text(json.dumps(value), encoding='utf-8')
    config = root / 'config.yaml'
    config.write_text('model: mock-model\n', encoding='utf-8')
    (root / 'mock-model').mkdir()
    (root / 'mock-model' / 'config.json').write_text('{}\n', encoding='utf-8')
    if failure == 'local_model': (root / 'mock-model' / 'config.json').unlink()
    secret = 'OFFLINE_SECRET_SENTINEL'
    cfg = {'out_path': root / 'out', 'model': 'mock-model', 'device': 'cpu',
           'dtype': 'float32', 'api_key': secret, 'nested': {'hf_token': secret}}
    args = SimpleNamespace(config=str(config), vectors_dir=str(source if failure != 'source_path' else root / 'other'))
    monkeypatch.setattr(script, 'ROOT', root)
    monkeypatch.setattr(script, 'KIT', kit)
    monkeypatch.setattr(script, 'REGISTERED_HASHES', {name: script.sha256(source / name) for name in script.REGISTERED_HASHES})
    if failure == 'source_hash': monkeypatch.setitem(script.REGISTERED_HASHES, 'directions.pt', '0' * 64)
    monkeypatch.setattr(script, 'setup', lambda *a, **k: (cfg, None, None, None))
    monkeypatch.setattr(script, '_git', lambda *a: 'offline-commit' if a[0] == 'rev-parse' else '')
    monkeypatch.setattr(script.sys, 'argv', ['decode', '--config', str(config), '--vectors-dir', str(source)])
    monkeypatch.setattr(script, 'load_trait', lambda name, split: copy.deepcopy(traits[name]))
    monkeypatch.setattr(script, 'chat_prompt', lambda tok, q: q)
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=22, hidden_size=8),
        generation_config=SimpleNamespace(eos_token_id=[2, 3],
            to_dict=lambda: {'eos_token_id': [2, 3], 'use_cache': True}))
    model_calls, generation_calls, score_calls = [], [], []
    def fake_model(*a):
        model_calls.append(a)
        return model, Tok()
    monkeypatch.setattr(script, 'load_model', fake_model)
    def fake_generate(model, tok, prompts, **kwargs):
        generation_calls.append((list(prompts), kwargs))
        short = kwargs['max_new_tokens'] == 16
        phase = kwargs['steer_phase']
        different = short and failure == 'preflight' and phase == 'decode'
        result = [generation([8 if different else 7, 2]) for _ in prompts]
        traces = phase_batches(len(prompts), phase)
        if failure == 'preflight_trace' and short and phase == 'decode': traces[0]['trace'][0]['applied'] = True
        if failure == 'phase_trace' and not short and phase == 'decode': traces[0]['trace'][0]['applied'] = True
        kwargs['phase_trace_out'].extend(traces)
        if not short and failure == 'token_text': result[0]['text'] = '与token不一致'
        if not short and failure == 'source_before_score': config.write_text('model: changed\n', encoding='utf-8')
        return result
    monkeypatch.setattr(script, 'generate', fake_generate)
    def score_many(prompts):
        assert len(generation_calls) == 68
        score_calls.append(prompts)
        if failure == 'secret_error': raise RuntimeError('远端错误 Authorization: Bearer ' + secret)
        if failure == 'source_changed': config.write_text('model: changed\n', encoding='utf-8')
        values = [prompt_value(p) for p in prompts]
        if failure == 'missing_score': values[0] = None
        return values
    def fake_judge(**kwargs):
        assert not Path(kwargs['cache_path']).exists()
        assert Path(kwargs['response_audit_path']).parent == Path(kwargs['cache_path']).parent
        return SimpleNamespace(model='changed' if failure == 'judge_changed' else 'deepseek-flash',
            base_url=f'https://user:{secret}@example.invalid/v1?api_key={secret}',
            max_tokens=16, extra_body={'authorization': secret}, score_many=score_many)
    monkeypatch.setattr(script, 'Judge', fake_judge)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            script.run(args)
        out = next((cfg['out_path'] / 'decode_strength').iterdir())
    else: out = script.run(args)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    meta_text = (out / 'run_meta.json').read_text(encoding='utf-8')
    meta = json.loads(meta_text)
    assert meta['status'] == ('failed' if failure else 'completed')
    assert meta['end_time'] and meta['formal_e0_pass'] is False and secret not in meta_text
    rows = [json.loads(line) for line in (out / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    if failure in ('source_path', 'source_hash', 'local_model', 'judge_changed'):
        assert not model_calls and not generation_calls and not score_calls
    elif failure in ('preflight', 'preflight_trace'):
        assert len(generation_calls) == 4 and not rows and not score_calls
        assert len(json.loads((out / 'preflight.json').read_text(encoding='utf-8'))) == 4
    elif failure in ('token_text', 'phase_trace'):
        assert not score_calls
        assert len(generation_calls) == (9 if failure == 'token_text' else 11)
        raw = [json.loads(line) for line in (out / 'generation_batches.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(raw) == (1 if failure == 'token_text' else 3)
        assert len(rows) == (0 if failure == 'token_text' else 60)
    else:
        assert len(generation_calls) == 68 and sum(len(prompts) for prompts, _ in generation_calls) == 1808
        assert len(rows) == 1800 and len(score_calls) == (0 if failure == 'source_before_score' else 1)
        script.validate_rows(rows, traits)
        if failure == 'missing_score': assert any(r[m] is None for r in rows for m in script.METRICS)
        if failure != 'source_before_score':
            plan = json.loads((out / 'score_plan.json').read_text(encoding='utf-8'))
            assert plan['n_inputs'] == 5400 and plan['n_unique'] == 120 and plan['seed'] == 0
        if not failure:
            assert all(r[m] == prompt_value((traits[r['target']]['eval_prompt'] if m == 'trait' else
                       getattr(script, m.upper() + '_PROMPT')).format(question=r['q'], answer=r['a']))
                       for r in rows for m in script.METRICS)
            assessment = json.loads((out / 'assessment.json').read_text(encoding='utf-8'))
            assert assessment['n_samples'] == 1800 and assessment['formal_e0_pass'] is False
            assert len(json.loads((out / 'preflight.json').read_text(encoding='utf-8'))) == 8
    if failure: assert not (out / 'assessment.json').exists()
    for prompts, kwargs in generation_calls:
        assert kwargs['seed'] in ((0,) if kwargs['max_new_tokens'] == 16 else (1, 2, 3))
        assert kwargs['batch_size'] == 16 and kwargs['temperature'] == kwargs['top_p'] == 1
        assert kwargs['return_details'] and kwargs['return_token_ids']
        assert not any('问题10' in p or '问题19' in p for p in prompts)
    manifest = json.loads((out / 'artifact_hashes.json').read_text(encoding='utf-8'))
    assert all(script.sha256(out / name) == digest for name, digest in manifest.items())
    with zipfile.ZipFile(out / 'results.zip') as archive:
        assert archive.testzip() is None and 'run_meta.json' in archive.namelist()
        assert all(secret.encode() not in archive.read(name) for name in archive.namelist())
