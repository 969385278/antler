"""C4：CPU微型模型检查串扰矩阵方向、基线复用及钩子清理。"""
import importlib
from pathlib import Path

import pytest
import torch

from test_core import model
from antler.core.hooks import Steer,no_steer
from antler.traits.data import DIMS
from antler.traits.data import BIG_FIVE


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


def test_signed_mode_is_opt_in_and_validates_all_coefficients(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    module=importlib.import_module('e0_crosstalk')
    module.validate_coefs([.2]*5)
    with pytest.raises(ValueError):module.validate_coefs([-.2,.2,.2,.2,.2])
    module.validate_coefs([-.2,.2,-.2,.2,.2],signed=True)
    for invalid in ([0,.2,.2,.2,.2], [float('nan')]*5, [float('inf')]*5, [.2]*4):
        with pytest.raises(ValueError):module.validate_coefs(invalid,signed=True)
    monkeypatch.setattr('sys.argv',['e0_crosstalk','--layer','1','--coefs','.2','.2','.2','.2','.2'])
    args=module.parse_args(module.add_arguments)
    assert args.measure_poles=='high' and args.signed_coefs is False


def test_signed_poles_select_low_vectors_and_cleanup_hooks(model,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    module=importlib.import_module('e0_crosstalk')
    eye=torch.eye(model.config.hidden_size)
    poles={name:eye[2*i+j] for i,dim in enumerate(DIMS) for j,name in enumerate(BIG_FIVE[dim])}
    coefs=[-.2,-.175,.125,-.2,.165]
    vectors=module.build_interventions(coefs,'pole',1,torch.zeros(2,5,model.config.hidden_size),torch.tensor([1.,2.]),poles)
    for i,dim in enumerate(DIMS):
        expected=eye[2*i+(coefs[i]<0)]*(2*abs(coefs[i]))
        assert torch.allclose(vectors[dim][1],expected)
        with torch.no_grad(),Steer(model,vectors[dim]):model(input_ids=torch.tensor([[1,2,3]]))
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_both_poles_use_separate_baselines_and_keep_matrix_orientation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    module=importlib.import_module('e0_crosstalk')
    vectors={d:object() for d in DIMS};calls=[];saved=[]
    def measure(dim,steer,pole):
        calls.append((dim,steer,pole))
        base=80 if pole=='high' else 5
        effect=10 if pole=='high' else 3
        return {'trait':base if steer is None else base+effect+DIMS.index(dim)}
    report=module.measure_pole_matrices(measure,vectors,saved.append,'both')
    assert len(calls)==60 and sum(s is None for _,s,_ in calls)==10
    assert report['by_pole']['high']['matrix']==[[10+i]*5 for i in range(5)]
    assert report['by_pole']['low']['matrix']==[[3+i]*5 for i in range(5)]
    assert sum(r['measured_pole']=='low' for r in saved)==30
    assert not report['formal_e0_pass']
    default=module.measure_pole_matrices(measure,vectors,lambda r:None)
    assert 'matrix' in default and 'by_pole' not in default
