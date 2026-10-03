"""批量生成文本。"""
from __future__ import annotations

import torch

from .hooks import Steer, no_steer
from .model import model_device


@torch.no_grad()
def generate(model, tok, prompts: list[str], *, steer: dict[int, torch.Tensor] | None = None,
             max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0,
             batch_size: int = 16, seed: int | None = None,
             return_details: bool = False) -> list[str] | list[dict]:
    """prompts 是已经套好对话模板的字符串。返回每条 prompt 生成的回答文本。

    temperature=0 表示贪心解码。steer 不为 None 时，整个生成过程都注入这些向量。
    return_details=True 时返回 text、n_generated_tokens、finish_reason。
    token 数包括首个 EOS，不包括提示或停止后的批补齐；恰好在长度上限输出 EOS 也记为 eos。
    """
    if seed is not None:
        torch.manual_seed(seed)
    tok.padding_side = "left"
    device = model_device(model)
    outs = []
    # 使用模型实际生成配置；tokenizer 的 EOS 不一定是 generate 的停止条件。
    generation_config = getattr(model, "generation_config", getattr(model, "config", None))
    eos_ids = getattr(generation_config, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids or [])
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
        sequences = gen.sequences if hasattr(gen, "sequences") else gen
        new_tokens = sequences[:, enc["input_ids"].shape[1]:]
        if not return_details:
            outs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
            continue
        for token_ids in new_tokens.tolist():
            # 先找 EOS，再截掉批补齐；pad==eos 时不能先按 pad 去尾。
            eos_at = next((j for j, token in enumerate(token_ids) if token in eos_ids), None)
            count = len(token_ids) if eos_at is None else eos_at + 1
            text = tok.batch_decode([token_ids[:count]], skip_special_tokens=True)[0]
            outs.append({"text": text, "n_generated_tokens": count,
                         "finish_reason": "length" if eos_at is None else "eos"})
    return outs
