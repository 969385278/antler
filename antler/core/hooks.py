"""向残差流注入向量（Steer）和读取残差流激活（Recorder）。

两者都通过 PyTorch forward hook 实现，挂在第 ℓ 个 decoder 层的输出上：
    h_ℓ ← h_ℓ + vec
用 with 语句使用，退出时钩子自动卸载，模型恢复原样——这就是“可拆卸”。
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .model import get_layers


def _split(out):
    """decoder 层的输出可能是张量，也可能是 (hidden, ...) 元组。"""
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


def _join(h, rest):
    return (h,) + rest if rest is not None else h


class Steer:
    """在一个或多个层的残差流上加向量。

    vectors: {层号: 向量 (d,)}。向量可以带梯度（蒸馏法优化 δ 时需要）。
    向量会加到该层输出的所有 token 位置上（包括 prompt 和生成的回答）。
    """

    def __init__(self, model, vectors: dict[int, torch.Tensor]):
        self.model = model
        self.vectors = vectors
        self._handles = []

    def _make_hook(self, vec):
        def hook(module, inputs, out):
            h, rest = _split(out)
            h = h + vec.to(device=h.device, dtype=h.dtype)
            return _join(h, rest)
        return hook

    def __enter__(self):
        layers = get_layers(self.model)
        for idx, vec in self.vectors.items():
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(vec)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class Recorder:
    """记录指定层的输出激活。每次前向后 self.acts[层号] = (batch, seq, d)。"""

    def __init__(self, model, layers: list[int] | None = None):
        self.model = model
        all_layers = get_layers(model)
        self.layer_ids = list(range(len(all_layers))) if layers is None else list(layers)
        self.acts: dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        all_layers = get_layers(self.model)
        for idx in self.layer_ids:
            def hook(module, inputs, out, idx=idx):
                h, _ = _split(out)
                self.acts[idx] = h.detach()
            self._handles.append(all_layers[idx].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


@contextmanager
def no_steer():
    """占位：表示不注入，方便写 `with (Steer(...) if v is not None else no_steer()):`"""
    yield None
