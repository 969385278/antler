"""C4：CPU微型模型检查串扰矩阵方向、基线复用及钩子清理。"""
import importlib
from pathlib import Path

import pytest
import torch

from test_core import model
from antler.core.hooks import Steer,no_steer
from antler.traits.data import DIMS


def test_matrix_orientation_and_baseline_reuse(model,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    module=importlib.import_module('e0_crosstalk')
    vectors={d:{1:torch.eye(model.config.hidden_size)[i]*(i+1)} for i,d in enumerate(DIMS)}
    calls=[];saved=[]
    def measure(dim,steer):
        calls.append((dim,steer))
        with torch.no_grad(),(Steer(model,steer) if steer else no_steer()):model(input_ids=torch.tensor([[1,2,3]]))
        i=DIMS.index(dim)
        j=next((DIMS.index(d) for d in DIMS if vectors[d] is steer),None)
        return {'trait':10+i+(0 if j is None else 10*(i+1)+(j+1))}
    result=module.measure_matrix(measure,vectors,saved.append)
    assert len(calls)==30 and sum(s is None for _,s in calls)==5
    assert result['matrix']==[[10*(i+1)+(j+1) for j in range(5)] for i in range(5)]
    assert result['max_abs_offdiag_over_mean_abs_diag']==pytest.approx(54/33)
    assert len(saved)==30 and not result['formal_e0_pass']
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_zero_diagonal_has_no_ratio(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    module=importlib.import_module('e0_crosstalk')
    result=module.measure_matrix(lambda *a:{'trait':50},{d:None for d in DIMS},lambda r:None)
    assert result['matrix']==[[0]*5 for _ in DIMS]
    assert result['max_abs_offdiag_over_mean_abs_diag'] is None
