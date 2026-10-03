"""向残差流注入向量（Steer）和读取残差流激活（Recorder）。

两者都通过 PyTorch forward hook 实现，挂在第 ℓ 个 decoder 层的输出上：
    h_ℓ ← h_ℓ + vec
用 with 语句使用，退出时钩子自动卸载，模型恢复原样——这就是“可拆卸”。
"""
from __future__ import annotations

from contextlib import contextmanager
from numbers import Integral

import torch
from transformers.cache_utils import Cache

from .model import get_layers


STEER_PHASES = ("all", "prefill", "decode")


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
    默认向量加到该层输出的所有 token 位置上（包括 prompt 和生成的回答）。
    phase=prefill/decode 时，仅在空缓存/已有缓存的顶层 forward 中注入。
    decode 不改变由首次 prefill 产生的首个输出 token；非默认模式需要有效 KV cache。
    直接调用时首次非空缓存视为 decode；同一上下文之后不能清空缓存，新批次须新建上下文。
    """

    def __init__(self, model, vectors: dict[int, torch.Tensor], *, phase: str = "all"):
        if phase not in STEER_PHASES:
            raise ValueError("phase 必须是 all、prefill 或 decode")
        self.model = model
        self.vectors = vectors
        self.phase = phase
        self.phase_trace: list[dict] = []
        self._active_phase = None
        self._seen_forward = False
        self._handles = []

    def _begin_forward(self, module, args, kwargs):
        """在 Cache 被各 decoder 层原地更新前，读取明确的缓存元数据。"""
        if self._active_phase is not None:
            raise ValueError("阶段注入不支持重入的顶层 forward")
        if len(args) > 1:
            raise ValueError("阶段注入要求缓存及 use_cache 通过关键字传入")
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = getattr(getattr(module, "config", None), "use_cache", None)
        if use_cache is not True:
            raise ValueError("prefill/decode 阶段注入需要 use_cache=True")
        cache = kwargs.get("past_key_values")
        if cache is None:
            cached = 0
        elif isinstance(cache, Cache):
            try:
                cached = cache.get_seq_length()
            except (TypeError, ValueError, NotImplementedError) as exc:
                raise ValueError("缓存不能提供有效的已缓存位置数") from exc
        else:
            raise ValueError("阶段注入不支持此 past_key_values 缓存类型")
        if isinstance(cached, torch.Tensor):
            if cached.numel() != 1:
                raise ValueError("已缓存位置数必须是非负整数标量")
            cached = cached.item()
        if isinstance(cached, bool) or not isinstance(cached, Integral) or cached < 0:
            raise ValueError("已缓存位置数必须是非负整数标量")
        cached = int(cached)
        if cached == 0 and self._seen_forward:
            raise ValueError("同一阶段注入上下文不能清空缓存；新批次需要新建 Steer 上下文")
        positions = kwargs.get("cache_position")
        first = None
        if positions is not None:
            if (not isinstance(positions, torch.Tensor) or positions.ndim != 1 or positions.numel() == 0
                    or positions.dtype not in (torch.int32, torch.int64)):
                raise ValueError("cache_position 必须是非空的一维整数张量")
            first = int(positions[0].item())
            if first != cached or not torch.equal(
                    positions, torch.arange(cached, cached+positions.numel(), device=positions.device,
                                            dtype=positions.dtype)):
                raise ValueError("cache_position 与已缓存位置数冲突或不连续")
        phase = "prefill" if cached == 0 else "decode"
        self._active_phase = phase
        self._seen_forward = True
        self.phase_trace.append({"phase": phase, "cached_tokens": cached,
                                 "cache_position_start": first,
                                 "applied": phase == self.phase and bool(self.vectors)})

    def _finish_forward(self, module, args, out):
        # always_call 保证 decoder 或顶层 forward 抛错时也清空当前阶段。
        self._active_phase = None

    def _make_hook(self, vec):
        def hook(module, inputs, out):
            if self.phase != "all":
                if self._active_phase is None:
                    raise ValueError("阶段注入必须通过模型顶层 forward 调用")
                if self._active_phase != self.phase:
                    return out
            h, rest = _split(out)
            h = h + vec.to(device=h.device, dtype=h.dtype)
            return _join(h, rest)
        return hook

    def __enter__(self):
        if self._handles:
            raise ValueError("同一个 Steer 上下文不能重复进入")
        self._active_phase = None
        self._seen_forward = False
        self.phase_trace = []
        layers = get_layers(self.model)
        try:
            if self.phase != "all":
                self._handles.append(self.model.register_forward_pre_hook(self._begin_forward, with_kwargs=True))
                self._handles.append(self.model.register_forward_hook(self._finish_forward, always_call=True))
            for idx, vec in self.vectors.items():
                self._handles.append(layers[idx].register_forward_hook(self._make_hook(vec)))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._active_phase = None
        self._seen_forward = False
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
