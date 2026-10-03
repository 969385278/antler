"""C19阶段诊断离线测试：固定身份、停止闸门、评分映射与结果留档。"""
from collections import Counter
import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import zipfile

import pytest
import torch


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e0_phase_diagnostics')


@pytest.fixture
def traits():
    return {p: {'questions': [f'{p}问题{i}' for i in range(20)],
                'eval_prompt': p + '\n问题：{question}\n回答：{answer}'}
            for p in ('dependable', 'careless', 'nervous', 'calm')}


class Tok:
    def batch_decode(self, rows, **kwargs):
        return [' '.join(str(t) for t in row if t not in (2, 3)) for row in rows]


def generation(tokens=None, reason='eos'):
    tokens = [7, 2] if tokens is None else tokens
    return {'token_ids': tokens, 'text': Tok().batch_decode([tokens])[0],
            'n_generated_tokens': len(tokens), 'finish_reason': reason}


def make_rows(traits):
    rows = []
    for dim, poles, coef in [('C', ('dependable', 'careless'), -.175),
                             ('N', ('nervous', 'calm'), .165)]:
        for pole, target in zip(('high', 'low'), poles):
            for arm in ('zero', 'all', 'prefill', 'decode'):
                for sample in range(3):
                    for q in range(10):
                        rows.append({'dim': dim, 'pole': pole, 'target': target, 'arm': arm,
                                     'question_index': q, 'sample_index': sample,
                                     'q': traits[target]['questions'][q], 'a': '7',
                                     'layer': 21, 'coef': 0 if arm == 'zero' else coef,
                                     'axis': 'pole', 'split': 'calib', 'seed': 0,
                                     'token_ids': [7, 2], 'n_generated_tokens': 2,
                                     'finish_reason': 'eos'})
    return rows


def make_vectors(path):
    (path / 'vectors').mkdir(parents=True)
    directions = {'U': torch.eye(5, 8).repeat(22, 1, 1),
                  's': torch.full((22,), 3.), 'dims': ['O', 'C', 'E', 'A', 'N']}
    torch.save(directions, path / 'directions.pt')
    for index, pole in enumerate(('dependable', 'careless', 'nervous', 'calm')):
        vector = torch.zeros(22, 8)
        vector[:, index] = index + 1
        torch.save(vector, path / 'vectors' / f'{pole}.pt')
    return directions


def test_complete_design_contains_only_480_calib_identities(script, traits):
    rows = make_rows(traits)
    script.validate_rows(rows, traits)
    assert len(rows) == 480
    assert Counter((r['dim'], r['pole'], r['arm']) for r in rows) == {
        (d, p, a): 30 for d in ('C', 'N') for p in ('high', 'low')
        for a in ('zero', 'all', 'prefill', 'decode')}
    assert {r['question_index'] for r in rows} == set(range(10))


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'test_index', 'wrong_question',
                                   'wrong_target', 'split', 'seed', 'layer', 'axis', 'coef'])
def test_invalid_identity_or_protocol_is_rejected(script, traits, change):
    rows = make_rows(traits)
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    elif change == 'test_index':
        rows[0]['question_index'] = 10
        rows[0]['q'] = traits['dependable']['questions'][10]
    elif change == 'wrong_question':
        rows[0]['q'] = '来源不符的问题'
    elif change == 'wrong_target':
        rows[0]['target'] = 'careless'
    else:
        rows[0][change] = {'split': 'test', 'seed': 1, 'layer': 20,
                           'axis': 'bipolar', 'coef': -.175}[change]
    with pytest.raises(ValueError):
        script.validate_rows(rows, traits)


def test_vectors_use_correct_pole_scale_and_fixed_layer(script, tmp_path):
    make_vectors(tmp_path / 'paired')
    vectors, shape, files = script.load_vectors(tmp_path / 'paired')
    assert shape == (22, 5, 8) and len(files) == 5
    assert vectors['C'].norm().item() == pytest.approx(.175 * 3)
    assert vectors['N'].norm().item() == pytest.approx(.165 * 3)
    assert torch.count_nonzero(vectors['C']).item() == 1
    assert vectors['C'][1] > 0 and vectors['N'][2] > 0


@pytest.mark.parametrize('change', ['layers', 'dims', 'scale', 'shape', 'nan', 'zero'])
def test_invalid_vectors_fail_before_any_model_load(script, tmp_path, change):
    path = tmp_path / 'paired'
    d = make_vectors(path)
    if change == 'layers':
        d['U'], d['s'] = d['U'][:21], d['s'][:21]
    elif change == 'dims':
        d['dims'] = list(reversed(d['dims']))
    elif change == 'scale':
        d['s'][21] = 0
    if change in ('layers', 'dims', 'scale'):
        torch.save(d, path / 'directions.pt')
    else:
        vector = torch.ones(22, 8)
        if change == 'shape': vector = vector[:, :7]
        if change == 'nan': vector[21, 0] = float('nan')
        if change == 'zero': vector[21] = 0
        torch.save(vector, path / 'vectors' / 'careless.pt')
    with pytest.raises(ValueError):
        script.load_vectors(path)


def test_generation_accepts_eos_at_limit_and_counts_all_eos_ids(script):
    script.validate_generation([generation([7, 3])], 1, 2, {2, 3})
    script.validate_generation([generation([7, 8], 'length')], 1, 2, {2, 3})


@pytest.mark.parametrize('change', ['count', 'over_limit', 'short_length', 'early_eos',
                                   'reason', 'empty_text', 'negative_token', 'bool_token'])
def test_generation_rejects_inconsistent_token_eos_records(script, change):
    item = generation([7, 8, 2])
    if change == 'count': item['n_generated_tokens'] = 2
    if change == 'over_limit': item = generation([7, 8, 9, 2])
    if change == 'short_length': item = generation([7, 8], 'length')
    if change == 'early_eos': item['token_ids'][0] = 2
    if change == 'reason': item['finish_reason'] = 'length'
    if change == 'empty_text': item['text'] = ' '
    if change == 'negative_token': item['token_ids'][0] = -1
    if change == 'bool_token': item['token_ids'][0] = True
    with pytest.raises(ValueError):
        script.validate_generation([item], 1, 3, {2})
    with pytest.raises(ValueError):
        script.validate_generation([], 1, 3, {2})


@pytest.mark.parametrize('mismatch', [False, True])
def test_preflight_has_eight_zero_vector_generations_and_stops_on_difference(
        script, traits, monkeypatch, mismatch):
    calls = []
    def fake_generate(model, tok, prompts, **kwargs):
        calls.append(kwargs)
        different = mismatch and kwargs.get('steer_phase') == 'decode'
        return [generation([8 if different else 7, 2])]
    monkeypatch.setattr(script, 'generate', fake_generate)
    monkeypatch.setattr(script, 'chat_prompt', lambda tok, q: q)
    vectors = {'C': torch.ones(8), 'N': torch.full((8,), 2.)}
    if mismatch:
        with pytest.raises(ValueError, match='零向量'):
            script.preflight(object(), Tok(), vectors, traits, {2})
        assert len(calls) == 4
    else:
        records = script.preflight(object(), Tok(), vectors, traits, {2})
        assert len(records) == len(calls) == 8
    for call in calls:
        assert call['max_new_tokens'] == 16 and call['seed'] == 0
        if call['steer'] is not None:
            assert set(call['steer']) == {21}
            assert torch.count_nonzero(call['steer'][21]).item() == 0


def prompt_value(prompt):
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 101


def test_scoring_shuffle_full_prompt_dedup_and_mapping(script, traits, tmp_path):
    rows = [{'target': target, 'q': q, 'a': a} for target, q, a in [
        ('careless', '同一问题', '同一回答'), ('careless', '同一问题', '同一回答'),
        ('nervous', '同一问题', '同一回答'), ('careless', '另一问题', '同一回答')]]
    calls = []
    judge = SimpleNamespace(score_many=lambda prompts: calls.append(prompts) or
                            [prompt_value(p) for p in prompts])
    assert script.score_rows(judge, rows, traits, tmp_path)
    jobs = []
    for index, row in enumerate(rows):
        for metric, template in [('trait', traits[row['target']]['eval_prompt']),
                                 ('coherence', script.COHERENCE_PROMPT), ('task', script.TASK_PROMPT)]:
            prompt = template.format(question=row['q'], answer=row['a'])
            jobs.append((index, metric, prompt))
            assert row[metric] == prompt_value(prompt)
    random.Random(0).shuffle(jobs)
    assert len(calls) == 1
    assert calls[0] == list(dict.fromkeys(p for _, _, p in jobs))
    assert len(calls[0]) == 7
    plan = json.loads((tmp_path / 'score_plan.json').read_text(encoding='utf-8'))
    assert plan['seed'] == 0 and plan['n_inputs'] == 12 and plan['n_unique'] == 7
    assert plan['ordered_inputs'] == [{'row': i, 'metric': m,
                                      'sha256': hashlib.sha256(p.encode()).hexdigest()}
                                     for i, m, p in jobs]
    # 再次构造同一计划只用于离线确定性测试，不对真实评委追加请求。
    script.score_rows(judge, rows, traits, tmp_path)
    assert calls[0] == calls[1]


def test_missing_scores_remain_none_and_incomplete_count_rejected(script, traits, tmp_path):
    rows = [{'target': 'careless', 'q': '问题', 'a': '回答'}]
    judge = SimpleNamespace(score_many=lambda prompts: [None] + [88] * (len(prompts)-1))
    assert script.score_rows(judge, rows, traits, tmp_path) is False
    assert sum(rows[0][m] is None for m in script.METRICS) == 1
    assert all(rows[0][m] in (None, 88) for m in script.METRICS)
    with pytest.raises(ValueError, match='条数'):
        script.score_rows(SimpleNamespace(score_many=lambda prompts: []), rows, traits, tmp_path)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 101, True, '80'])
def test_invalid_score_is_rejected_before_mapping(script, traits, tmp_path, value):
    rows = [{'target': 'careless', 'q': '问题', 'a': '回答'}]
    judge = SimpleNamespace(score_many=lambda prompts: [value] * len(prompts))
    with pytest.raises(ValueError):
        script.score_rows(judge, rows, traits, tmp_path)
    assert all(metric not in rows[0] for metric in script.METRICS)


def test_analysis_weights_questions_and_compares_each_arm_to_all(script, traits):
    rows = make_rows(traits)
    for row in rows:
        row['trait'] = {'zero': 0, 'all': 10, 'decode': 5,
                        'prefill': 40 if row['question_index'] == 0 else 20}[row['arm']]
        row['coherence'] = row['task'] = 90
    # 增加同一题的重复数；点估计仍应对10道题等权，而非对回答数加权。
    rows += [copy.deepcopy(r) for r in rows if r['question_index'] == 0] * 3
    result = script.analyze(rows)
    assert result['formal_e0_pass'] is False and len(result['conditions']) == 12
    for condition in result['conditions']:
        zero = condition['versus_zero']['metrics']['trait']
        assert zero['n_questions'] == 10
        if condition['arm'] == 'prefill':
            assert zero['mean'] == 22
            assert condition['versus_all']['trait']['mean'] == 12
        if condition['arm'] == 'decode':
            assert condition['versus_all']['trait']['mean'] == -5


@pytest.mark.parametrize('failure', [None, 'preflight', 'missing_score', 'secret_error',
                                    'token_text', 'source_changed', 'judge_changed'])
def test_mock_run_preserves_sources_and_records_success_or_failure(
        script, traits, tmp_path, monkeypatch, failure):
    root = tmp_path / 'repo'
    source = root / 'paired'
    make_vectors(source)
    (root / 'antler').mkdir()
    (root / 'antler' / 'synthetic.py').write_text('# 合成来源\n', encoding='utf-8')
    kit = root / 'kit'
    (kit / 'trait_eval').mkdir(parents=True)
    for name, value in traits.items():
        (kit / 'trait_eval' / f'{name}.json').write_text(json.dumps(value), encoding='utf-8')
    config = root / 'config.yaml'
    config.write_text('model: mock-model\n', encoding='utf-8')
    secret = 'OFFLINE_SECRET_SENTINEL'
    cfg = {'out_path': root / 'out', 'model': 'mock-model', 'device': 'cpu',
           'dtype': 'float32', 'api_key': secret, 'nested': {'hf_token': secret}}
    args = SimpleNamespace(config=str(config), vectors_dir=str(source))
    monkeypatch.setattr(script, 'ROOT', root)
    monkeypatch.setattr(script, 'KIT', kit)
    monkeypatch.setattr(script, 'setup', lambda *a, **k: (cfg, None, None, None))
    monkeypatch.setattr(script, '_git', lambda *a: 'offline-commit' if a[0] == 'rev-parse' else '')
    monkeypatch.setattr(script.sys, 'argv', ['phase', '--config', str(config), '--vectors-dir', str(source)])
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
        different = short and failure == 'preflight' and kwargs.get('steer_phase') == 'decode'
        result = [generation([8 if different else 7, 2]) for _ in prompts]
        if not short and failure == 'token_text':
            result[0]['text'] = '与token不一致'
        return result
    monkeypatch.setattr(script, 'generate', fake_generate)
    def score_many(prompts):
        score_calls.append(prompts)
        if failure == 'secret_error':
            raise RuntimeError('远端错误 Authorization: Bearer ' + secret)
        if failure == 'source_changed':
            config.write_text('model: changed\n', encoding='utf-8')
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
        out = next((cfg['out_path'] / 'phase_diagnostics').iterdir())
    else:
        out = script.run(args)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    meta_text = (out / 'run_meta.json').read_text(encoding='utf-8')
    meta = json.loads(meta_text)
    assert meta['status'] == ('failed' if failure else 'completed')
    assert meta['end_time'] and meta['formal_e0_pass'] is False
    assert secret not in meta_text
    rows = [json.loads(line) for line in (out / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    if failure == 'judge_changed':
        assert not model_calls and not generation_calls and not score_calls
    elif failure == 'preflight':
        assert len(generation_calls) == 4 and not rows and not score_calls
    elif failure == 'token_text':
        assert len(generation_calls) == 9 and not rows and not score_calls
    else:
        assert len(generation_calls) == 24
        assert sum(len(prompts) for prompts, _ in generation_calls) == 488
        assert len(rows) == 480 and len(score_calls) == 1
        script.validate_rows(rows, traits)
        if failure == 'missing_score':
            assert any(r[m] is None for r in rows for m in script.METRICS)
        if not failure:
            assessment = json.loads((out / 'assessment.json').read_text(encoding='utf-8'))
            assert assessment['formal_e0_pass'] is False
            assert len(json.loads((out / 'preflight.json').read_text(encoding='utf-8'))) == 8
    if failure:
        assert not (out / 'assessment.json').exists()
    for prompts, kwargs in generation_calls:
        assert kwargs['seed'] == 0 and kwargs['batch_size'] == 16
        assert kwargs['temperature'] == kwargs['top_p'] == 1
        assert kwargs['return_details'] and kwargs['return_token_ids']
        assert not any('问题10' in p or '问题19' in p for p in prompts)
    manifest = json.loads((out / 'artifact_hashes.json').read_text(encoding='utf-8'))
    assert all(script.sha256(out / name) == digest for name, digest in manifest.items())
    with zipfile.ZipFile(out / 'results.zip') as archive:
        assert archive.testzip() is None and 'run_meta.json' in archive.namelist()
        assert all(secret.encode() not in archive.read(name) for name in archive.namelist())
