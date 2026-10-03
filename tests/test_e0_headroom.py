"""C21：CPU离线验证余量起点、采样、Steer和来源记录。"""
import argparse
import importlib
import json
from pathlib import Path

import pytest
import torch

from test_core import model
from antler.core.hooks import Steer, no_steer
from antler.traits.data import DIMS, KIT
from antler.traits.headroom_prompts import HEADROOM_PROMPTS


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e0_headroom')


@pytest.mark.parametrize('prompt_id', list(HEADROOM_PROMPTS))
def test_headroom_flow(module, model, monkeypatch, tmp_path, prompt_id):
    dim = HEADROOM_PROMPTS[prompt_id]['dim']
    config = tmp_path / 'config.yaml'; config.write_text('model: mock\n', encoding='utf-8')
    cfg = {'out_path': tmp_path, 'model': 'mock', 'api_key': '不得进入记录'}
    torch.save({'U': torch.eye(5, model.config.hidden_size).repeat(4, 1, 1), 's': torch.ones(4), 'dims': DIMS}, tmp_path / 'directions.pt')
    monkeypatch.setattr(module, 'setup', lambda *a, **kw: (cfg, None, None, None))
    monkeypatch.setattr(module, 'load_model', lambda *a: (model, None))
    monkeypatch.setattr(module, 'Judge', lambda **kw: argparse.Namespace(model='固定测试评委'))
    calls = []

    def measure(m, tok, judge, d, **kw):
        calls.append(kw)
        assert d == dim and kw['system'] == HEADROOM_PROMPTS[prompt_id]['text']
        assert kw['n_questions'] == 10 and kw['n_samples'] == 3 and kw['max_new_tokens'] == 320
        with torch.no_grad(), (Steer(m, kw['steer']) if kw['steer'] else no_steer()):
            m(input_ids=torch.tensor([[1, 2, 3]]))
        return {'trait': 50., 'coherence': 90., 'task': 90., 'samples': []}

    monkeypatch.setattr(module, 'trait_score', measure)
    args = argparse.Namespace(config=str(config), dim=dim, prompt_id=prompt_id,
                              coefs=[0., .2] if dim == 'C' else [0., -.2], layer=1, axis='bipolar', n_samples=3)
    out = module.run(args)
    assert len(calls) == (2 if dim == 'C' else 4)
    assert sum(c['steer'] is None for c in calls) == (1 if dim == 'C' else 2)
    raw = (out / 'run_meta.json').read_text(encoding='utf-8')
    meta = json.loads(raw)
    assert meta['status'] == 'completed' and meta['prompt_id'] == prompt_id
    assert '不得进入记录' not in raw
    assert len(meta['artifact_hashes']) == 2
    assert all(not l._forward_hooks for l in model.model.layers)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_prompts_are_not_copied_from_extraction():
    texts = '\n'.join(p.read_text(encoding='utf-8') for p in (KIT / 'trait_extract').glob('*.json'))
    assert len(HEADROOM_PROMPTS) == 4
    for item in HEADROOM_PROMPTS.values():
        assert item['text'] not in texts


@pytest.mark.parametrize('dim,prompt,coefs', [('C','n_sensitive',[0,.2]), ('N','n_sensitive',[0,.2]),
                                           ('C','c_easygoing',[.2]), ('C','c_easygoing',[0,float('nan')])])
def test_invalid_settings(module, dim, prompt, coefs):
    with pytest.raises(ValueError):
        module.validate(argparse.Namespace(dim=dim, prompt_id=prompt, coefs=coefs, n_samples=3), {})
