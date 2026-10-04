"""C10：独立标注输入隔离、严格解析、请求归档及密钥不落盘。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antler.eval.episode_annotation import parse_annotation, request_text


def test_prompt_excludes_expected_labels():
    assert 'A short story' in request_text({'id': 'blind1', 'text': 'A short story'})
    with pytest.raises(ValueError):
        request_text({'id': 'blind1', 'text': 'story', 'intended': {'O': 1}})


@pytest.mark.parametrize('text', ['{"labels":{"O":true},"reason":"x"}',
                                 '{"labels":{"O":0},"reason":"x"}',
                                 '{"labels":{"O":1,"O":-1},"reason":"x"}',
                                 '{"labels":{},"reason":""}', 'not json'])
def test_bad_labels_rejected(text):
    with pytest.raises(ValueError):parse_annotation(text)


def test_valid_neutral_and_fenced_reply():
    assert parse_annotation('```json\n{"labels":{},"reason":"No supported change"}\n```')['labels'] == {}


def test_explicit_plus_is_opt_in_and_preserves_strings():
    value = '{"labels":{"O":+1,"N":-1},"reason":"literal :+1, and \\"O\\":+1 remain"}'
    # 使用json编码字符串，避免测试本身产生非法转义。
    reason = 'literal :+1, and "O":+1 remain'
    value = '{"labels":{"O":+1,"N":-1},"reason":' + json.dumps(reason) + '}'
    with pytest.raises(ValueError): parse_annotation(value)
    result = parse_annotation(value, allow_explicit_plus=True)
    assert result == {'labels': {'O': 1, 'N': -1}, 'reason': reason}
    for bad in ('+10', '+1.0', '++1', 'true', '"+1"'):
        with pytest.raises(ValueError):
            parse_annotation('{"labels":{"O":' + bad + '},"reason":"x"}', allow_explicit_plus=True)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    module = importlib.import_module('d1_annotate')
    packet = tmp_path / 'blind.jsonl'
    packet.write_text(json.dumps({'id': 'D1B0001', 'text': 'Only the blind story.'}) + '\n')
    env = tmp_path / '.env'
    env.write_text('MINIMAX_API_KEY=private-test-secret\nMINIMAX_BASE_URL=https://example.org/v1\nMINIMAX_MODEL=test-model\n')
    for key in ('MINIMAX_API_KEY', 'MINIMAX_BASE_URL', 'MINIMAX_MODEL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(module, 'setup', lambda *a, **k: ({'out_path': tmp_path / 'out'}, None, None, None))
    args = SimpleNamespace(packet=str(packet), provider='minimax', env_file=str(env), config='unused')
    return module, args, tmp_path


def test_request_saved_without_key_and_labels_match_builder(case):
    module, args, root = case
    calls = []
    def post(url, **kw):
        calls.append(kw)
        assert kw['allow_redirects'] is False
        assert kw['json']['thinking'] == {'type': 'disabled'}
        assert kw['headers']['Authorization'] == 'Bearer private-test-secret'
        return SimpleNamespace(status_code=200, json=lambda: {'model': 'test-model', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': '{"labels":{"E":1},"reason":"Outcome supports more contact"}'}}]})
    out = module.run(args, post=post)
    assert len(calls) == 1
    for p in out.iterdir():
        assert 'private-test-secret' not in p.read_text(encoding='utf-8')
    labels = json.loads((out / 'labels.json').read_text())
    assert labels['records'][0]['labels'] == {'E': 1}
    assert labels['input_sha256'] == module.sha256(args.packet)
    assert json.loads((out / 'run_meta.json').read_text())['status'] == 'completed'


def test_failed_request_not_retried_or_exception_secret_saved(case):
    module, args, root = case
    calls = []
    def post(*a, **kw):
        calls.append(1)
        raise RuntimeError('private-test-secret')
    with pytest.raises(RuntimeError) as exc:
        module.run(args, post=post)
    assert 'private-test-secret' not in str(exc.value) and len(calls) == 1
    metas = list((root / 'out').glob('d1_annotations/*/run_meta.json'))
    assert json.loads(metas[0].read_text())['status'] == 'failed'


def test_configuration_rejects_embedded_credentials(case):
    module, args, root = case
    Path(args.env_file).write_text('MINIMAX_API_KEY=x\nMINIMAX_MODEL=m\nMINIMAX_BASE_URL=https://user:key@example.org/v1\n')
    with pytest.raises(ValueError, match='HTTPS'):
        module.read_connection('minimax', args.env_file)


def test_resume_reuses_archived_responses_and_only_requests_suffix(case):
    module, args, root = case
    Path(args.packet).write_text(''.join(json.dumps({'id': f'D1B{i:04d}', 'text': f'Story {i}.'}) + '\n' for i in range(3)))
    calls = []
    def reply(content):
        return SimpleNamespace(status_code=200, json=lambda: {'model': 'test-model', 'choices': [
            {'finish_reason': 'stop', 'message': {'content': content}}]})
    def initial(*a, **k):
        calls.append(1)
        return reply('{"labels":{"O":' + ('1' if len(calls) == 1 else '+1') + '},"reason":"x"}')
    with pytest.raises(ValueError): module.run(args, post=initial)
    assert len(calls) == 2
    source = next((root / 'out' / 'd1_annotations').iterdir())
    saved = {p.name: p.read_bytes() for p in source.iterdir()}
    args.resume = str(source); args.allow_explicit_plus = True
    remaining = []
    def suffix(*a, **k):
        remaining.append(k['json'])
        return reply('{"labels":{},"reason":"no change"}')
    out = module.run(args, post=suffix)
    assert len(remaining) == 1
    assert all((source / name).read_bytes() == content for name, content in saved.items())
    labels = json.loads((out / 'labels.json').read_text())['records']
    assert len(labels) == 3 and labels[1]['explicit_plus_normalized']
    assert not labels[0]['explicit_plus_normalized']
    assert (out / '0001_response.json').read_bytes() == saved['0001_response.json']
    with pytest.raises(FileExistsError): module.run(args, post=suffix)
    assert len(remaining) == 1


def test_unknown_request_cannot_be_retried_via_resume(case):
    module, args, root = case
    def failure(*a, **k): raise RuntimeError('network')
    with pytest.raises(RuntimeError): module.run(args, post=failure)
    source = next((root / 'out' / 'd1_annotations').iterdir())
    args.resume = str(source); args.allow_explicit_plus = True
    with pytest.raises(ValueError, match='未知'):
        module.run(args, post=lambda *a, **k: pytest.fail('不应发出请求'))
