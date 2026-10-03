"""把一段经历写成人格位移 δ ∈ R^5（快系统的“写入”）。

经历以“记忆”的形式放进系统提示（老师），对比没有这段记忆时（基线），测量或蒸馏出它对人格的影响：

方法 A（投影法，免训练）：
    δ = solve(U Uᵀ, U mean_p [a_ℓ(p | 经历) − a_ℓ(p | 中性记忆)] / s_ℓ)
  默认用对偶基恢复非正交方向的坐标；projection="dot"保留旧点积法作为对照。
  a_ℓ 取最后一个 prompt token 的激活。得到的方向较可信，但大小 ≠ 注入时需要的大小。

方法 B（蒸馏法，只优化 5 个数）：
    min_δ  mean_p KL( M(·|经历, p) ‖ M_δ(·|p) ) + λ‖δ‖²
  学生模型没有这段记忆（只有一段中性记忆作对照），只注入 s_ℓ·(δ·U)。在老师生成的回答 token 上逐位置算 KL（上下文蒸馏）。
  方向和大小一起确定，用方法 A 的结果做初始化。
"""
from __future__ import annotations

import math
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
# 对照组用的中性记忆：格式和实验组完全相同，内容与人格无关。
# 若对照组是“没有记忆”，测到的主要是“系统提示里多了一段记忆”这一格式效应（debug 运行中所有经历的 δ_A 几乎相同）。
# 多条中性记忆，结构与试验经历对齐（请求 → 回应 → 用户反馈 → 结果），长度相近，内容与人格无关。
# 对照组取它们的平均，避免“有情节的故事 vs 一句流水账”这种形式差异混进测量。
NEUTRAL_MEMORIES = [
    "A user asked me to convert a recipe from cups to grams. I listed the conversions, they checked them "
    "against their kitchen scale, said the numbers matched, and went back to baking for the afternoon.",
    "Someone asked me which time zone Lisbon is in before a work call. I told them, they added the meeting "
    "to their calendar, thanked me briefly, and the call took place as planned the next morning.",
    "A student asked me to format their bibliography in APA style. I reformatted the twelve entries, they "
    "pasted them into their paper, and later mentioned that the assignment had been submitted on time.",
    "A user asked me for the boiling point of water at high altitude. I explained the approximate values, "
    "they noted them down for an upcoming camping trip, and said it was the information they needed.",
    "Someone asked me to translate a short train timetable from German. I translated the departure times, "
    "they confirmed the platform with a station clerk, and caught their usual train to work that day.",
]
NEUTRAL_MEMORY = NEUTRAL_MEMORIES[0]  # 兼容旧代码


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


def _baseline_prompts(tok, probes, baseline: str) -> list[str]:
    """对照组 prompt，与 probes 一一对应。
    'neutral'（默认）：第 i 道探针配第 i % 5 条中性记忆，整体上是 5 条中性记忆的平均；
    'none'：不放任何记忆（会混入格式效应，仅用于对比）。"""
    if baseline == "none":
        return _prompts(tok, probes, None)
    return [chat_prompt(tok, p, system=MEMORY_TEMPLATE.format(experience=NEUTRAL_MEMORIES[i % len(NEUTRAL_MEMORIES)]))
            for i, p in enumerate(probes)]


def projection_coordinates(activation_difference: torch.Tensor, U_layer: torch.Tensor,
                           s_layer: float, *, projection: str = "dual") -> torch.Tensor:
    """C7：在五行满秩基中恢复最小二乘坐标；dot保留旧点积对照。

    用float64求解五维Gram方程，不显式求逆，不通过正则项偷偷改变坐标定义。
    秩不足时对偶坐标不唯一，明确失败。输出沿用float32及U所在设备。
    """
    if projection not in ("dual", "dot"):
        raise ValueError("投影方法必须为dual或dot")
    if (not isinstance(U_layer, torch.Tensor) or U_layer.ndim != 2
            or U_layer.shape[0] != len(DIMS) or U_layer.shape[1] < len(DIMS)
            or not U_layer.is_floating_point() or not torch.isfinite(U_layer).all()):
        raise ValueError("人格基须为有限浮点数的5×d矩阵，且d至少为5")
    if (not isinstance(activation_difference, torch.Tensor) or activation_difference.ndim != 1
            or activation_difference.shape[0] != U_layer.shape[1]
            or not activation_difference.is_floating_point()
            or not torch.isfinite(activation_difference).all()):
        raise ValueError("激活差须为与人格基隐藏维度一致的有限浮点向量")
    if type(s_layer) not in (int, float) or not math.isfinite(s_layer) or s_layer <= 0:
        raise ValueError("层尺度必须为有限正数")
    if projection == "dot":
        # 显式对照保持旧float32点积行为，不把它当成非正交基的真实坐标。
        delta = U_layer.float() @ activation_difference.to(U_layer.device, torch.float32) / s_layer
        if not torch.isfinite(delta).all():
            raise ValueError("点积坐标产生非有限值")
        return delta
    basis = U_layer.to(dtype=torch.float64)
    if int(torch.linalg.matrix_rank(basis)) != len(DIMS):
        raise ValueError("人格基不满秩，无法唯一恢复五维对偶坐标")
    difference = activation_difference.to(device=basis.device, dtype=basis.dtype)
    delta = torch.linalg.solve(basis @ basis.T, basis @ (difference / s_layer))
    if not torch.isfinite(delta).all():
        raise ValueError("对偶坐标求解产生非有限值")
    result = delta.to(dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("对偶坐标转换为float32后产生非有限值")
    return result


def delta_projection(model, tok, experience: str, probes: list[str], U_layer: torch.Tensor,
                     s_layer: float, layer: int, baseline: str = "neutral", *,
                     projection: str = "dual") -> torch.Tensor:
    """方法 A。U_layer: (5, d) 第 ℓ 层的人格方向。返回 δ (5,)。"""
    if baseline not in ("neutral", "none"):
        raise ValueError("对照组必须为neutral或none")
    if not probes:
        raise ValueError("写入探针不能为空")
    a_with = last_prompt_acts(model, tok, _prompts(tok, probes, experience), layers=[layer])[:, 0]
    if baseline == "neutral":  # 每道探针都和全部 5 条中性记忆比较，取平均，降低噪声
        a_without = torch.stack([
            last_prompt_acts(model, tok, [chat_prompt(tok, p, system=MEMORY_TEMPLATE.format(experience=m))
                                          for p in probes], layers=[layer])[:, 0]
            for m in NEUTRAL_MEMORIES]).mean(0)
    else:
        a_without = last_prompt_acts(model, tok, _prompts(tok, probes, None), layers=[layer])[:, 0]
    d = (a_with - a_without).mean(0)                     # (d,)
    return projection_coordinates(d, U_layer, s_layer, projection=projection)


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
                  seed: int = 0, log_every: int = 10, baseline: str = "neutral") -> tuple[torch.Tensor, list[dict]]:
    """方法 B。返回 (δ, 训练日志)。"""
    device = model_device(model)
    with_prompts = _prompts(tok, probes, experience)
    without_prompts = _baseline_prompts(tok, probes, baseline)
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
def kl_without_steer(model, tok, experience: str, probes: list[str], max_new_tokens: int = 64,
                     baseline: str = "neutral") -> float:
    """参照值：完全不注入时，学生和老师的 KL。蒸馏后的 KL 应明显低于它。"""
    device = model_device(model)
    with_prompts = _prompts(tok, probes, experience)
    without_prompts = _baseline_prompts(tok, probes, baseline)
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
