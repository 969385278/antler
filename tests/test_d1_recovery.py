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
