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
