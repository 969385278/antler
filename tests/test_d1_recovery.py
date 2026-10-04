"""C10：补写不重试接口失败、不替换已有故事、保留第二次尝试。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antler.memory.d1_plan import make_plan
from antler.memory.episode_data import normalize
from test_episode_data import candidate


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('d1_recover')


def test_rate_limit_stops_without_retry(module, tmp_path):
    calls, waits = [], []
    def generate(slot, conf, out, **kwargs):
        calls.append(slot['id']); (out / slot['id']).mkdir()
        assert kwargs['attempt'] == 2
        return {'id': slot['id'], 'valid': False, 'reason': 'HTTP 429'}
    results = module.collect(make_plan()['slots'][:3], {}, tmp_path, set(), generate=generate, sleep=waits.append)
    assert len(results) == len(calls) == 1 and not waits
    assert (tmp_path / calls[0] / 'result.json').exists()


def test_duplicates_rejected_and_requests_spaced(module, tmp_path):
    waits = []
    def generate(slot, conf, out, **kwargs):
        (out / slot['id']).mkdir()
        return {'id': slot['id'], 'valid': True, 'candidate': candidate()}
    results = module.collect(make_plan()['slots'][:2], {}, tmp_path,
                             {normalize(candidate()['text'])}, generate=generate, sleep=waits.append)
    assert waits == [5]
    assert all(not r['valid'] and '重复' in r['reason'] for r in results)


def test_http_diagnostic_redacts_key(module, tmp_path):
    conf = {'MODEL': 'MiniMax-M3', 'endpoint': 'https://example.org', 'API_KEY': 'test-secret'}
    response = SimpleNamespace(status_code=429, headers={'Retry-After': '60'},
                               json=lambda: {'error': {'code': 'rate_limit', 'message': 'test-secret'}})
    slot = make_plan()['slots'][0]
    result = module.generate_one(slot, conf, tmp_path, lambda *a, **k: response)
    diagnostic = (tmp_path / slot['id'] / 'http_error.json').read_text()
    assert result['reason'] == 'HTTP 429'
    assert 'test-secret' not in diagnostic
    assert json.loads(diagnostic)['retry_after'] == '60'


def test_second_attempt_changes_prompt_and_provenance_only(module, tmp_path):
    conf = {'MODEL': 'MiniMax-M3', 'endpoint': 'https://example.org', 'API_KEY': 'test-secret'}
    slot = make_plan()['slots'][0]
    def post(url, **kwargs):
        assert module.SUFFIX in kwargs['json']['messages'][0]['content']
        return SimpleNamespace(status_code=200, json=lambda: {'model': conf['MODEL'], 'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps({'text': candidate()['text']})}}]})
    r = module.generate_one(slot, conf, tmp_path, post, prompt_suffix=module.SUFFIX, attempt=2)
    assert r['valid'] and r['candidate']['attempt'] == 2
    for k in ('id', 'intended', 'group_id', 'split'):
        assert r['candidate'][k] == slot[k]


def parent_fixture(module, tmp_path):
    parent = tmp_path / 'parent'; parent.mkdir()
    source = tmp_path / 'first'; source.mkdir()
    plan = make_plan(); first, second, third = plan['slots'][:3]
    original = [{**candidate(third['id'], third['intended']), **third}]
    module.write_json(parent / 'run_meta.json', {'status': 'stopped', 'source': str(source), 'completed_requests': 1})
    module.write_json(parent / 'split_plan.json', plan)
    module.write_json(parent / 'request_order.json', [first['id'], second['id']])
    module.write_json(parent / 'unattempted.json', [second['id']])
    (parent / first['id']).mkdir()
    module.write_json(parent / first['id'] / 'request.json', {})
    module.write_json(parent / first['id'] / 'result.json', {'valid': False})
    (parent / 'candidates.jsonl').write_text(json.dumps(original[0]) + '\n')
    return parent, source, plan, original


def seal(module, parent):
    module.write_json(parent / 'artifact_hashes.json', {str(p.relative_to(parent)): module.sha256(p)
        for p in parent.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


def test_continuation_only_takes_unrequested_suffix(module, tmp_path):
    parent, source, plan, original = parent_fixture(module, tmp_path)
    seal(module, parent)
    rows, slots = module.continuation(parent, source, plan, original)
    assert rows == original and slots == [plan['slots'][1]]


@pytest.mark.parametrize('fault', ['requested', 'reordered', 'replaced'])
def test_continuation_rejects_retry_or_replacement(module, tmp_path, fault):
    parent, source, plan, original = parent_fixture(module, tmp_path)
    if fault == 'requested': (parent / plan['slots'][1]['id']).mkdir()
    elif fault == 'reordered': module.write_json(parent / 'unattempted.json', [plan['slots'][0]['id']])
    else: (parent / 'candidates.jsonl').write_text('')
    seal(module, parent)
    with pytest.raises(ValueError): module.continuation(parent, source, plan, original)


def test_digital_role_is_explicit_opt_in(module, tmp_path):
    def generate(slot, conf, out, **kwargs):
        (out / slot['id']).mkdir()
        assert module.ROLE_SUFFIX in kwargs['prompt_suffix']
        return {'id': slot['id'], 'valid': True, 'candidate': candidate()}
    module.collect(make_plan()['slots'][:1], {}, tmp_path, set(), generate=generate, digital_role=True)
