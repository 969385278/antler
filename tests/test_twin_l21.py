"""C12：第21层独立入口和四单元低成本诊断。"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from antler.memory.twin_risk import decompose, geometry
from antler.memory.twin_state import analyze
from test_core import model


def test_factorial_reconstruction():
    h = torch.randn(4, 9, generator=torch.Generator().manual_seed(3))
    d = decompose(h)
    for row, (a, r) in zip(h, ((-1,-1),(-1,1),(1,-1),(1,1))):
        assert torch.allclose(row.double(), d['mean']+a*d['action']+r*d['feedback']+a*r*d['interaction'])


def test_story_permutation_has_four_not_thousand_independent_units():
    x = torch.zeros(9, 60, 8)
    x[1,:,0] = 2; x[3,:,0] = 2
    report = geometry(x, 2)
    assert report['terms']['interaction']['norm_s'] == 0
    assert report['unique_relabelings'] == 4
    assert report['formal_p_value'] is None
    assert report['probe_half_cosines']['interaction'] is None


def test_layer_summary_not_silently_sixteen():
    x = torch.zeros(9,60,8)
    assert analyze(x, torch.eye(5,8), 1)['layer'] == 16
    assert analyze(x, torch.eye(5,8), 1, layer=21)['layer'] == 21


def test_l21_preflight_no_generation(monkeypatch):
    import e1_twin_state_extract as entry
    monkeypatch.setattr(entry, 'setup', lambda *a, **k: ({},None,None,None))
    monkeypatch.setattr(entry, 'validate', lambda a: ([],[],None,None))
    monkeypatch.setattr(entry, 'load_model', lambda *a: pytest.fail('预检不加载模型'))
    assert entry.run(SimpleNamespace(execute=False, layer21_pilot=True))['protocol']['layer'] == 21


def test_real_tiny_model_layer_readout(model):
    from antler.core.hooks import Recorder
    from antler.core.model import get_layers
    with Recorder(model, [1, 2]) as recorder, torch.no_grad():
        model(input_ids=torch.tensor([[1,2,3]]))
    assert not torch.equal(recorder.acts[1], recorder.acts[2])
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not layer._forward_hooks for layer in get_layers(model))
