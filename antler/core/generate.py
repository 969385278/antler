"""批量生成文本。"""
from __future__ import annotations

import torch

from .hooks import Steer, no_steer
from .model import model_device


@torch.no_grad()
def generate(model, tok, prompts: list[str], *, steer: dict[int, torch.Tensor] | None = None,
             max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0,
             batch_size: int = 16, seed: int | None = None) -> list[str]:
    """prompts 是已经套好对话模板的字符串。返回每条 prompt 生成的回答文本。

    temperature=0 表示贪心解码。steer 不为 None 时，整个生成过程都注入这些向量。
    """
    if seed is not None:
        torch.manual_seed(seed)
    tok.padding_side = "left"
    device = model_device(model)
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        with (Steer(model, steer) if steer else no_steer()):
            gen = model.generate(**enc, **kwargs)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
    return outs
