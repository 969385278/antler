"""把一段经历写成人格位移 δ ∈ R^5（快系统的“写入”）。

经历以“记忆”的形式放进系统提示（老师），对比没有这段记忆时（基线），测量或蒸馏出它对人格的影响：

方法 A（投影法，免训练）：
    δ_k = mean_p [ a_ℓ(p | 经历) − a_ℓ(p) ] · u_k / s_ℓ
  a_ℓ 取最后一个 prompt token 的激活。得到的方向较可信，但大小 ≠ 注入时需要的大小。

方法 B（蒸馏法，只优化 5 个数）：
    min_δ  mean_p KL( M(·|经历, p) ‖ M_δ(·|p) ) + λ‖δ‖²
  学生模型没有这段记忆，只注入 s_ℓ·(δ·U)。在老师生成的回答 token 上逐位置算 KL（上下文蒸馏）。
  方向和大小一起确定，用方法 A 的结果做初始化。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..core.activations import last_prompt_acts
from ..core.generate import generate
from ..core.hooks import Steer
from ..core.model import chat_prompt, model_device
from ..traits.data import BIG_FIVE, DIMS, load_trait
from ..traits.vectors import steer_vector

BASE_SYSTEM = "You are an AI companion who talks with people every day."
MEMORY_TEMPLATE = BASE_SYSTEM + "\n\nSomething that happened to you recently, which you still remember:\n{experience}"


@dataclass
class Episode:
    id: str
    text: str                                   # 原文，只用于审计和删除，不参与注入
    intended: dict = field(default_factory=dict)  # 数据生成时标注的预期影响，如 {"C": +1, "O": -1}；只用于评测
    delta: torch.Tensor | None = None           # 人格位移 (5,)，单位是 s_ℓ 的比例
    salience: float = 1.0
    t: int = 0


def probe_questions(n: int = 30, seed: int = 0) -> list[str]:
    """写入用的探针：取五个维度高极点的“提取集”题目（与评测集不重叠），随机抽 n 道。"""
    qs = [q for k in DIMS for q in load_trait(BIG_FIVE[k][0], "extract")["questions"]]
    random.Random(seed).shuffle(qs)
    return qs[:n]


def _prompts(tok, probes, experience: str | None):
    system = MEMORY_TEMPLATE.format(experience=experience) if experience else BASE_SYSTEM
    return [chat_prompt(tok, p, system=system) for p in probes]


def delta_projection(model, tok, experience: str, probes: list[str], U_layer: torch.Tensor,
                     s_layer: float, layer: int) -> torch.Tensor:
    """方法 A。U_layer: (5, d) 第 ℓ 层的人格方向。返回 δ (5,)。"""
    a_with = last_prompt_acts(model, tok, _prompts(tok, probes, experience), layers=[layer])[:, 0]
    a_without = last_prompt_acts(model, tok, _prompts(tok, probes, None), layers=[layer])[:, 0]
    d = (a_with - a_without).mean(0)                     # (d,)
    return (U_layer.float() @ d) / s_layer


def _seq_logprobs(model, prompt_ids, resp_ids, steer=None):
    """在 prompt+response 上前向，返回预测 response 各 token 的 log 概率分布 (T_resp, V)。"""
    ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
    if steer is not None:
        with Steer(model, steer):
            logits = model(input_ids=ids).logits[0]
    else:
        logits = model(input_ids=ids).logits[0]
    start = prompt_ids.shape[0]
    return F.log_softmax(logits[start - 1: start - 1 + resp_ids.shape[0]].float(), dim=-1)


def delta_distill(model, tok, experience: str, probes: list[str], U_layer: torch.Tensor, s_layer: float,
                  layer: int, *, init: torch.Tensor | None = None, steps: int = 60, lr: float = 0.02,
                  lam: float = 1e-3, probes_per_step: int = 4, max_new_tokens: int = 64,
                  seed: int = 0, log_every: int = 10) -> tuple[torch.Tensor, list[dict]]:
    """方法 B。返回 (δ, 训练日志)。"""
    device = model_device(model)
    with_prompts = _prompts(tok, probes, experience)
    without_prompts = _prompts(tok, probes, None)
    # 老师（读了经历的模型）先生成回答，固定下来作为蒸馏的 token 序列
    teacher_resp = generate(model, tok, with_prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    enc = lambda s: torch.tensor(tok(s, add_special_tokens=False)["input_ids"], device=device)
    items = []
    for wp, wop, r in zip(with_prompts, without_prompts, teacher_resp):
        r_ids = enc(r)
        if r_ids.numel() == 0:
            continue
        with torch.no_grad():
            t_logp = _seq_logprobs(model, enc(wp), r_ids)  # 老师分布，固定
        items.append((enc(wop), r_ids, t_logp.cpu()))

    U = U_layer.to(device=device, dtype=torch.float32)
    delta = (init.clone().float() if init is not None else torch.zeros(len(DIMS))).to(device).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    rng = random.Random(seed)
    log = []
    for step in range(steps):
        batch = rng.sample(items, min(probes_per_step, len(items)))
        opt.zero_grad()
        total = 0.0
        for wop_ids, r_ids, t_logp in batch:
            vec = steer_vector(U, s_layer, delta)
            s_logp = _seq_logprobs(model, wop_ids, r_ids, steer={layer: vec})
            t = t_logp.to(device)
            kl = (t.exp() * (t - s_logp)).sum(-1).mean()   # KL(老师 ‖ 学生)，按 token 平均
            loss = kl / len(batch)
            loss.backward()
            total += float(kl)
        (lam * delta.pow(2).sum()).backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            log.append({"step": step, "kl": total / len(batch), "delta": delta.detach().cpu().tolist()})
    return delta.detach().cpu(), log


@torch.no_grad()
def kl_without_steer(model, tok, experience: str, probes: list[str], max_new_tokens: int = 64) -> float:
    """参照值：完全不注入时，学生和老师的 KL。蒸馏后的 KL 应明显低于它。"""
    device = model_device(model)
    with_prompts = _prompts(tok, probes, experience)
    without_prompts = _prompts(tok, probes, None)
    resp = generate(model, tok, with_prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    enc = lambda s: torch.tensor(tok(s, add_special_tokens=False)["input_ids"], device=device)
    kls = []
    for wp, wop, r in zip(with_prompts, without_prompts, resp):
        r_ids = enc(r)
        if r_ids.numel() == 0:
            continue
        t = _seq_logprobs(model, enc(wp), r_ids)
        s = _seq_logprobs(model, enc(wop), r_ids)
        kls.append(float((t.exp() * (t - s)).sum(-1).mean()))
    return sum(kls) / len(kls)
