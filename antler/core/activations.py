"""从模型里读激活：回答部分的平均激活、最后一个 prompt token 的激活、各层的典型范数 s_ℓ。

所有返回值都是 float32、放在 CPU 上，形状 (N, L, d)，L 为所请求的层数（按 layers 顺序）。
"""
from __future__ import annotations

import torch

from .hooks import Steer, Recorder, no_steer
from .model import model_device, n_layers


def _layers(model, layers):
    return list(range(n_layers(model))) if layers is None else list(layers)


@torch.no_grad()
def response_mean_acts(model, tok, prompts: list[str], responses: list[str], *,
                       layers: list[int] | None = None, batch_size: int = 8,
                       steer: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
    """对每对 (prompt, response)，取 response 各 token 在各层激活的平均。

    Persona Vectors 发现用回答 token（而不是 prompt token）提取的方向更有效。
    空回答对应的行是 NaN，调用方需要过滤。
    """
    assert len(prompts) == len(responses)
    layers = _layers(model, layers)
    device = model_device(model)
    tok.padding_side = "right"  # 右填充，回答的位置从 prompt 长度开始
    results = []
    for i in range(0, len(prompts), batch_size):
        ps, rs = prompts[i:i + batch_size], responses[i:i + batch_size]
        p_lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in ps]
        enc = tok([p + r for p, r in zip(ps, rs)], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with (Steer(model, steer) if steer else no_steer()), Recorder(model, layers) as rec:  # 先注入后记录：记录到的是注入后的激活
            model(**enc)
        total = enc["attention_mask"].sum(1).tolist()
        batch_out = torch.full((len(ps), len(layers), model.config.hidden_size), float("nan"))
        for b in range(len(ps)):
            start, end = p_lens[b], int(total[b])
            if end <= start:
                continue
            for j, l in enumerate(layers):
                batch_out[b, j] = rec.acts[l][b, start:end].float().mean(0).cpu()
        results.append(batch_out)
    tok.padding_side = "left"
    return torch.cat(results)


@torch.no_grad()
def last_prompt_acts(model, tok, prompts: list[str], *, layers: list[int] | None = None,
                     batch_size: int = 16, steer: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
    """取最后一个 prompt token（助手即将开口的位置）在各层的激活。

    Persona Vectors 用它来预测接下来回答的人格表达（r≈0.75–0.83），不需要生成，速度快。
    """
    layers = _layers(model, layers)
    device = model_device(model)
    tok.padding_side = "left"  # 左填充，最后一个位置就是 prompt 末尾
    results = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with (Steer(model, steer) if steer else no_steer()), Recorder(model, layers) as rec:  # 先注入后记录：记录到的是注入后的激活
            model(**enc)
        results.append(torch.stack([rec.acts[l][:, -1].float().cpu() for l in layers], dim=1))
    return torch.cat(results)


@torch.no_grad()
def layer_norms(model, tok, texts: list[str], *, batch_size: int = 8, skip_first: int = 1) -> torch.Tensor:
    """各层激活的平均 L2 范数 s_ℓ，形状 (n_layers,)。

    所有注入系数都以 s_ℓ 为单位：系数 0.1 表示加上“该层典型范数 10% 长度”的向量。
    跳过每条文本开头的 skip_first 个 token：Qwen 等模型的首个 token 激活范数异常大（attention sink），
    会严重拉高平均值。
    """
    L = n_layers(model)
    device = model_device(model)
    tok.padding_side = "right"
    sums = torch.zeros(L, dtype=torch.float64)
    count = 0
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False, truncation=True, max_length=512).to(device)
        with Recorder(model, None) as rec:
            model(**enc)
        mask = enc["attention_mask"].bool().clone()
        mask[:, :skip_first] = False
        for l in range(L):
            norms = rec.acts[l].float().norm(dim=-1)  # (B, T)
            sums[l] += norms[mask].double().sum().cpu()
        count += int(mask.sum())
    tok.padding_side = "left"
    return (sums / max(count, 1)).float()
