"""核心机制测试：用随机初始化的微型 Qwen2 模型，不需要下载任何东西，CPU 几秒跑完。

运行：pytest -q tests
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from antler.core.hooks import Recorder, Steer  # noqa: E402
from antler.core.model import freeze, get_layers  # noqa: E402
from antler.traits.vectors import build_directions, steer_vector  # noqa: E402
from antler.traits.data import BIG_FIVE  # noqa: E402

D, L = 32, 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=D, intermediate_size=64, num_hidden_layers=L,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    m = Qwen2ForCausalLM(cfg).eval()
    freeze(m)
    return m


@pytest.fixture
def ids():
    return torch.randint(0, 64, (2, 10))


def logits(m, ids):
    with torch.no_grad():
        return m(input_ids=ids).logits


def test_zero_vector_changes_nothing(model, ids):
    base = logits(model, ids)
    with Steer(model, {2: torch.zeros(D)}):
        assert torch.allclose(logits(model, ids), base, atol=1e-6)


def test_steer_changes_output_and_is_detachable(model, ids):
    base = logits(model, ids)
    with Steer(model, {2: torch.randn(D) * 5}):
        assert not torch.allclose(logits(model, ids), base, atol=1e-4)
    # 退出 with 之后钩子被卸载，模型恢复原样（“可拆卸”）
    assert torch.allclose(logits(model, ids), base, atol=1e-6)
    assert all(len(layer._forward_hooks) == 0 for layer in get_layers(model))


def test_recorder_sees_injected_vector(model, ids):
    vec = torch.randn(D)
    with Recorder(model, [1]) as r0:
        model(input_ids=ids)
    with Steer(model, {1: vec}), Recorder(model, [1]) as r1:
        model(input_ids=ids)
    # 同一层先注入后记录：记录值之差应正好等于注入的向量
    assert torch.allclose(r1.acts[1] - r0.acts[1], vec.expand_as(r0.acts[1]), atol=1e-5)
    assert r0.acts[1].shape == (2, 10, D)


def test_gradient_reaches_delta_but_not_model(model, ids):
    U = torch.nn.functional.normalize(torch.randn(5, D), dim=-1)
    delta = torch.zeros(5, requires_grad=True)
    with Steer(model, {2: steer_vector(U, 3.0, delta)}):
        loss = model(input_ids=ids).logits.pow(2).mean()
    loss.backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_steer_vector_math():
    U = torch.eye(5, D)
    z = torch.tensor([0.1, 0.0, -0.2, 0.0, 0.0])
    v = steer_vector(U, 10.0, z)
    assert torch.allclose(v[:5], torch.tensor([1.0, 0.0, -2.0, 0.0, 0.0]))


def test_build_directions_unit_norm():
    poles = {p: torch.randn(L, D) for pair in BIG_FIVE.values() for p in pair}
    U = build_directions(poles)
    assert U.shape == (L, 5, D)
    assert torch.allclose(U.norm(dim=-1), torch.ones(L, 5), atol=1e-5)
