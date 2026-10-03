"""C8：响应来源审计不改变请求和缓存语义，也不记录密钥或完整提示。"""
import json
from types import SimpleNamespace

import pytest

from antler.eval.judge import Judge, ResponseAuditError


def make_judge(tmp_path, monkeypatch, audit=True, missing=False):
    monkeypatch.delenv('JUDGE_EXTRA_BODY', raising=False)
    monkeypatch.setenv('JUDGE_MAX_TOKENS', '16')
    calls = []
    body = {'choices': [{'message': {'content': '85'}, 'finish_reason': 'stop'}]}
    if not missing:
        body.update(model='served-model', id='response-1', system_fingerprint='fp-1', created=123)
    def post(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(headers={'x-request-id': 'request-1', 'Authorization': 'secret'},
                               raise_for_status=lambda: None, json=lambda: body)
    monkeypatch.setattr('antler.eval.judge.requests.post', post)
    judge = Judge(str(tmp_path/'cache.jsonl'), base_url='https://example.invalid', api_key='private-key',
                  model='fixed-model', response_audit_path=str(tmp_path/'audit.jsonl') if audit else None)
    return judge, calls


def test_audit_preserves_request_and_cache(tmp_path, monkeypatch):
    judge, calls = make_judge(tmp_path, monkeypatch)
    assert judge.score_many(['完整测试提示', '完整测试提示']) == [85, 85]
    assert judge._call('完整测试提示') == '85'
    assert len(calls) == 1
    assert calls[0]['json'] == {'model': 'fixed-model', 'temperature': 0, 'max_tokens': 16,
                              'messages': [{'role': 'user', 'content': '完整测试提示'}]}
    text = (tmp_path/'audit.jsonl').read_text(encoding='utf-8')
    record = json.loads(text)
    assert record['returned_model'] == 'served-model'
    assert record['system_fingerprint'] == 'fp-1' and record['request_id'] == 'request-1'
    assert record['key'] == judge._key('完整测试提示')
    assert record['output'] == '85' and len(record['request_sha256']) == 64
    assert all(value not in text for value in ['完整测试提示', 'private-key', 'Authorization', 'secret'])
    assert json.loads((tmp_path/'cache.jsonl').read_text()) == {'key': record['key'], 'output': '85'}


def test_disabled_and_missing_identity(tmp_path, monkeypatch):
    judge, _ = make_judge(tmp_path, monkeypatch, audit=False)
    assert judge._call('题目') == '85'
    assert not (tmp_path/'audit.jsonl').exists()
    judge, _ = make_judge(tmp_path, monkeypatch, missing=True)
    assert judge._call('新题目') == '85'
    row = json.loads((tmp_path/'audit.jsonl').read_text())
    assert row['returned_model'] is None and row['system_fingerprint'] is None


def test_audit_failure_does_not_retry_remote_request(tmp_path, monkeypatch):
    judge, calls = make_judge(tmp_path, monkeypatch)
    (tmp_path/'audit.jsonl').mkdir()
    with pytest.raises(ResponseAuditError):
        judge._call('题目')
    assert len(calls) == 1
    assert not (tmp_path/'cache.jsonl').exists()


def test_reject_cache_audit_path_collision(tmp_path):
    with pytest.raises(ValueError, match='不能与评分缓存相同'):
        Judge(str(tmp_path/'same'), base_url='https://example.invalid', api_key='key', model='fixed',
              response_audit_path=str(tmp_path/'same'))
