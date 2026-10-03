"""C7：非正交坐标恢复与真实微型模型注入一致，无下载。"""
import pytest
import torch

from antler.memory import write
from antler.core.hooks import Recorder, Steer
from antler.core.model import get_layers
from test_core import model


def basis(d=9):
    u = torch.eye(5, d)
    u[1] = (u[0] * .8 + u[1] * .6)
    u[3] = (u[2] * -.6 + u[3] * .8)
    return u


def test_nonorthogonal_dual_recovers_known_coordinates_and_dot_is_gram_mixed():
    u = basis()
    z = torch.tensor([.2, -.3, .1, .4, -.2])
    difference = 7.5 * (z @ u)
    dual = write.projection_coordinates(difference, u, 7.5)
    dot = write.projection_coordinates(difference, u, 7.5, projection='dot')
    assert torch.allclose(dual, z, atol=1e-6)
    assert torch.allclose(dot, u @ u.T @ z, atol=1e-6)
    assert not torch.allclose(dot, z, atol=1e-3)


def test_orthogonal_basis_agrees_and_outside_span_is_not_invented_as_personality():
    u = torch.eye(5, 8)
    difference = torch.tensor([1., -2., 3., -4., 5., 6., -7., 8.])
    dual = write.projection_coordinates(difference, u, 2.)
    assert torch.equal(dual, write.projection_coordinates(difference, u, 2., projection='dot'))
    assert torch.equal(dual, difference[:5] / 2)
    assert torch.allclose(u @ (difference - 2 * (dual @ u)), torch.zeros(5))


def test_nonorthogonal_least_squares_residual_is_orthogonal_to_all_rows():
    u = basis()
    d = torch.arange(9, dtype=torch.float32)
    z = write.projection_coordinates(d, u, 3.)
    residual = d - 3 * (z @ u)
    assert torch.allclose(u @ residual, torch.zeros(5), atol=2e-6)


def test_rank_deficient_dual_fails_and_dot_remains_available():
    u = basis(); u[4] = u[3]
    with pytest.raises(ValueError, match='满秩'):
        write.projection_coordinates(torch.ones(9), u, 1.)
    assert write.projection_coordinates(torch.ones(9), u, 1., projection='dot').shape == (5,)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf'), True, '1'])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match='尺度'):
        write.projection_coordinates(torch.ones(9), basis(), scale)


@pytest.mark.parametrize('method', ['dual', 'dot'])
def test_finite_inputs_cannot_return_overflowed_float32_coordinates(method):
    with pytest.raises(ValueError, match='非有限值'):
        write.projection_coordinates(torch.ones(9), basis(), 1e-40, projection=method)


@pytest.mark.parametrize('which', ['bad_method', 'bad_rows', 'bad_width', 'nan_basis',
                                  'nan_difference', 'integer_basis', 'integer_difference', 'matrix_difference'])
def test_invalid_projection_inputs(which):
    u, d, method = basis(), torch.ones(9), 'dual'
    if which == 'bad_method': method = 'inverse'
    elif which == 'bad_rows': u = u[:4]
    elif which == 'bad_width': d = d[:8]
    elif which == 'nan_basis': u[0, 0] = float('nan')
    elif which == 'nan_difference': d[0] = float('nan')
    elif which == 'integer_basis': u = u.long()
    elif which == 'integer_difference': d = d.long()
    elif which == 'matrix_difference': d = d[None]
    with pytest.raises(ValueError):
        write.projection_coordinates(d, u, 1., projection=method)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_low_precision_basis_is_solved_in_supported_precision(dtype):
    u = basis().to(dtype)
    z = torch.tensor([1., 2., -1., .5, -.5])
    d = z @ u.float()
    found = write.projection_coordinates(d, u, 1.)
    assert found.dtype == torch.float32 and torch.allclose(found, z, atol=2e-6)


def test_delta_projection_uses_all_five_neutral_memories_and_default_dual(monkeypatch):
    u = basis()
    z = torch.tensor([.1, -.2, .3, -.4, .5])
    change = 2 * z @ u
    calls = []
    monkeypatch.setattr(write, 'chat_prompt', lambda tok, p, system: system + '|' + p)
    def acts(model, tok, prompts, layers):
        calls.append(prompts)
        mean = 10 + change if len(calls) == 1 else torch.full((9,), 10.)
        return mean.expand(len(prompts), 1, 9).clone()
    monkeypatch.setattr(write, 'last_prompt_acts', acts)
    found = write.delta_projection(None, None, '经历', ['探针甲', '探针乙'], u, 2., 1)
    assert len(calls) == 6 and all(len(x) == 2 for x in calls)
    for prompts, memory in zip(calls[1:], write.NEUTRAL_MEMORIES):
        assert all(memory in prompt for prompt in prompts)
    assert torch.allclose(found, z, atol=2e-6)


def test_tiny_model_observed_injection_recovers_coordinates_and_leaves_weights_frozen(model):
    torch.manual_seed(23)
    ids = torch.randint(0, 64, (2, 7))
    u, z, scale = basis(32), torch.tensor([.1, -.2, .3, -.4, .5]), 2.
    with torch.no_grad(), Recorder(model, [1]) as before:
        model(input_ids=ids)
    with torch.no_grad(), Steer(model, {1: scale * (z @ u)}), Recorder(model, [1]) as after:
        model(input_ids=ids)
    difference = (after.acts[1] - before.acts[1]).mean((0, 1))
    assert torch.allclose(write.projection_coordinates(difference, u, scale), z, atol=2e-6)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not layer._forward_hooks for layer in get_layers(model))
