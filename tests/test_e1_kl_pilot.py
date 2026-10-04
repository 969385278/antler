"""C11：真实token语料构造与可归档入口的CPU离线测试。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_core import model, D, L


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e1_kl_pilot')


class Tokenizer:
    def __call__(self, text, **kwargs):
        # 故意使文本分词与生成token不同，验证不重分词老师回答。
        return {'input_ids': [5, 6, 7]}

    def batch_decode(self, rows, **kwargs):
        return ['answer' for _ in rows]


def test_corpus_retains_actual_tokens_and_five_neutrals(module, model, monkeypatch):
    monkeypatch.setattr(module, 'chat_prompt', lambda tok, q, system: system + q)
    calls = []
    def generate(m, t, prompts, **kwargs):
        calls.append(kwargs)
        return [dict(text='answer', token_ids=[12, 2], n_generated_tokens=2, finish_reason='eos') for _ in prompts]
    monkeypatch.setattr(module, 'generate', generate)
    items, records = module.prepare_corpus(model, Tokenizer(), 'experience', ['a', 'b'], 16)
    assert len(calls) == 1 and calls[0]['return_token_ids'] is True
    assert all(i.response_ids.tolist() == [12, 2] for i in items)
    assert all(len(i.neutral_prompt_ids) == 5 and i.teacher_logp.shape == (2, 64) for i in items)
    assert len(set(records[0]['neutral_prompts'])) == 5


def test_preflight_does_not_replace_full_settings(module):
    small, full = module.settings(True), module.settings(False)
    assert (small['n_episodes'], small['steps']) == (1, 2)
    assert (full['n_episodes'], full['n_fit'], full['n_eval'], full['steps']) == (12, 20, 10, 60)
    q = module.random_basis(D)
    assert torch.equal(q, module.random_basis(D))
    assert torch.allclose(q @ q.T, torch.eye(5), atol=1e-6)


def test_entrypoint_archive_and_failure(module, model, monkeypatch, tmp_path):
    config = tmp_path / 'config.yaml'; config.write_text('model: unused\n')
    directions = tmp_path / 'directions.pt'; directions.write_bytes(b'fixture')
    episodes = tmp_path / 'episodes.jsonl'; episodes.write_text('fixture')
    model_dir = tmp_path / 'model'; model_dir.mkdir(); (model_dir / 'config.json').write_text('{}')
    cfg = {'model': str(model_dir), 'out_path': tmp_path / 'out'}
    monkeypatch.setattr(module, 'setup', lambda a, **kw: (cfg, None, None, None))
    monkeypatch.setattr(module, 'LAYER', 2)
    monkeypatch.setattr(module, 'load_inputs', lambda *a: (torch.eye(5, D).repeat(L, 1, 1), torch.ones(L),
                                                         [{'id': f'p{i:02}', 'text': 'story', 'intended': {'O': 1}} for i in range(1, 13)]))
    monkeypatch.setattr(module, 'load_model', lambda *a: (model, Tokenizer()))
    monkeypatch.setattr(module, 'probe_questions', lambda *a, **k: [f'probe{i}' for i in range(30)])
    monkeypatch.setattr(module, 'chat_prompt', lambda tok, q, system: system + q)
    monkeypatch.setattr(module, 'generate', lambda m, t, prompts, **kw: [dict(text='answer', token_ids=[12, 2], n_generated_tokens=2, finish_reason='eos') for _ in prompts])
    monkeypatch.setattr(module, 'fit_activation_difference', lambda *a: torch.zeros(D))
    fitted = []
    def fit(m, train, evaluation, **kw):
        fitted.append(([i.probe for i in train], [i.probe for i in evaluation], kw['basis']))
        return dict(baseline={'kl': .2}, final={'kl': .1}, coordinates=torch.zeros(5), vector=torch.zeros(D), explained=.5)
    monkeypatch.setattr(module, 'fit_and_evaluate', fit)
    args = SimpleNamespace(config=str(config), directions=str(directions), experiences=str(episodes), preflight=True)
    out = module.run(args)
    meta = json.loads((out / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed' and meta['episodes_completed'] == 1
    assert len(fitted) == 3
    assert all(a == ['probe0', 'probe1'] and b == ['probe20', 'probe21'] for a, b, _ in fitted)
    assert fitted[-1][-1] is None
    saved = torch.load(out / 'p01/teacher_corpus.pt', weights_only=True)
    assert len(saved) == 4 and saved[0]['response_ids'].tolist() == [12, 2]
    for name, digest in json.loads((out / 'artifact_hashes.json').read_text(encoding='utf-8')).items():
        assert module.sha256(out / name) == digest
    def fail(*a, **kw):
        raise RuntimeError('预期资源故障')
    monkeypatch.setattr(module, 'prepare_corpus', fail)
    with pytest.raises(RuntimeError, match='资源故障'):
        module.run(args)
    metas = [json.loads(p.read_text(encoding='utf-8')) for p in cfg['out_path'].glob('e1_kl_pilot/*/run_meta.json')]
    assert sorted(m['status'] for m in metas) == ['completed', 'failed']
