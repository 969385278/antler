"""人格向量的提取（Persona Vectors 方法）和人格坐标系。

提取一个极点（例如 dependable）的向量：
  1. 对提取集的每道题，分别用 5 条“诱发该特质”的系统提示（pos）和 5 条“抑制该特质”的系统提示（neg）生成回答；
  2. （可选）用评委过滤：默认独立保留 pos > 50、neg < 50；显式开启连贯性门槛时，
     成对保留 pos >= 50、neg < 50 且两侧连贯性都达标的回答；
  3. 取每条回答在各层的“回答 token 平均激活”，pos 组均值 − neg 组均值，得到每层一个向量。

人格坐标系：每个大五维度 k 的方向 u_k = normalize(v_高极 − v_低极)，形状 (n_layers, 5, d)。
"""
from __future__ import annotations

from pathlib import Path
import math

import torch

from ..core.activations import response_mean_acts
from ..core.generate import generate
from ..core.model import chat_prompt
from .data import BIG_FIVE, DIMS, load_trait


def extraction_rows(pole: str, *, n_rollouts: int = 2,
                    max_questions: int | None = None, samples: list[dict] | None = None) -> list[dict]:
    """按原提取顺序还原系统提示及配对位置；提供旧回答时严格检查数量和顺序。"""
    if isinstance(n_rollouts, bool) or not isinstance(n_rollouts, int) or n_rollouts < 1:
        raise ValueError("n_rollouts 必须是正整数")
    if max_questions is not None and (isinstance(max_questions, bool)
                                     or not isinstance(max_questions, int) or max_questions < 0):
        raise ValueError("max_questions 必须是非负整数或 None；0 保持旧版的全部题目行为")
    data = load_trait(pole, "extract")
    questions = data["questions"][:max_questions] if max_questions else data["questions"]
    rows = []
    for pair_index, pair in enumerate(data["instruction"]):
        for question_index, question in enumerate(questions):
            for rollout in range(n_rollouts):
                for side in ("pos", "neg"):
                    rows.append({"source_index": len(rows), "pair_index": pair_index,
                                 "question_index": question_index, "rollout": rollout,
                                 "side": side, "system": pair[side], "question": question})
    if samples is not None:
        if not isinstance(samples, list) or len(samples) != len(rows):
            raise ValueError(f"{pole}: 原始样本数量与提取配置不一致，应为 {len(rows)}")
        for row, sample in zip(rows, samples):
            if (not isinstance(sample, dict) or sample.get("side") != row["side"]
                    or sample.get("question") != row["question"]):
                raise ValueError(f"{pole}: source_index={row['source_index']} 的顺序、题目或 side 不一致")
    return rows


def select_paired_samples(samples: list[dict], coherence: list[float | None] | None = None,
                          *, min_coherence: float | None = None) -> tuple[list[int], list[int]]:
    """按相邻 pos/neg 成对筛选；None 仅关闭连贯性门槛，仍保留配对人格规则。"""
    if len(samples) % 2:
        raise ValueError("配对筛选需要偶数条 pos/neg 回答")
    if min_coherence is not None:
        if (isinstance(min_coherence, bool) or not isinstance(min_coherence, (int, float))
                or not math.isfinite(min_coherence) or not 0 <= min_coherence <= 100):
            raise ValueError("min_coherence 必须为 0–100 的有限数值或 None")
        if coherence is None or len(coherence) != len(samples):
            raise ValueError("连贯性评分数量与原始样本数不一致")
        if any(s is not None and (isinstance(s, bool) or not isinstance(s, (int, float))
                                 or not math.isfinite(s) or not 0 <= s <= 100) for s in coherence):
            raise ValueError("连贯性分数必须为 0–100 的有限数值或 None")
    pos, neg = [], []
    for i in range(0, len(samples), 2):
        a, b = samples[i:i + 2]
        if a.get("side") != "pos" or b.get("side") != "neg" or a.get("question") != b.get("question"):
            raise ValueError(f"source_index={i}: 回答没有按同题 pos/neg 配对")
        scores = (a.get("score"), b.get("score"))
        if any(s is not None and (isinstance(s, bool) or not isinstance(s, (int, float))
                                 or not math.isfinite(s) or not 0 <= s <= 100) for s in scores):
            raise ValueError("人格分数必须为 0–100 的有限数值或 None")
        if any(s is None for s in scores) or not (scores[0] >= 50 and scores[1] < 50):
            continue
        if min_coherence is not None:
            pair_scores = coherence[i:i + 2]
            if any(s is None or s < min_coherence for s in pair_scores):
                continue
        pos.append(i)
        neg.append(i + 1)
    return pos, neg


def extract_pole(model, tok, pole: str, *, judge=None, n_rollouts: int = 2, max_new_tokens: int = 200,
                 batch_size: int = 16, seed: int = 0, max_questions: int | None = None,
                 min_coherence: float | None = None) -> dict:
    if min_coherence is not None and judge is None:
        raise ValueError("开启连贯性过滤必须提供评委")
    if min_coherence is not None:
        select_paired_samples([], [], min_coherence=min_coherence)
    data = load_trait(pole, "extract")
    rows = [(r["side"], r["system"], r["question"])
            for r in extraction_rows(pole, n_rollouts=n_rollouts, max_questions=max_questions)]
    prompts = [chat_prompt(tok, q, system=s) for _, s, q in rows]
    answers = generate(model, tok, prompts, max_new_tokens=max_new_tokens, temperature=1.0,
                       batch_size=batch_size, seed=seed)

    scores = [None] * len(rows)
    if judge is not None:
        scores = judge.trait_scores(data["eval_prompt"], [q for _, _, q in rows], answers)

    def keep(i):
        side, s = rows[i][0], scores[i]
        if judge is None:
            return True
        if s is None:
            return False
        return s > 50 if side == "pos" else s < 50

    samples = [{"side": rows[i][0], "question": rows[i][2], "answer": answers[i], "score": scores[i]}
               for i in range(len(rows))]
    coherence = None
    if min_coherence is None:
        pos_idx = [i for i in range(len(rows)) if rows[i][0] == "pos" and keep(i)]
        neg_idx = [i for i in range(len(rows)) if rows[i][0] == "neg" and keep(i)]
    else:
        coherence = judge.coherence_scores([q for _, _, q in rows], answers)
        pos_idx, neg_idx = select_paired_samples(samples, coherence, min_coherence=min_coherence)
    if len(pos_idx) < 5 or len(neg_idx) < 5:
        raise RuntimeError(f"{pole}: 过滤后样本太少（pos={len(pos_idx)}, neg={len(neg_idx)}），检查评委或生成质量")

    acts_pos = response_mean_acts(model, tok, [prompts[i] for i in pos_idx], [answers[i] for i in pos_idx],
                                  batch_size=max(1, batch_size // 2))
    acts_neg = response_mean_acts(model, tok, [prompts[i] for i in neg_idx], [answers[i] for i in neg_idx],
                                  batch_size=max(1, batch_size // 2))
    vec = torch.nanmean(acts_pos, 0) - torch.nanmean(acts_neg, 0)  # (L, d)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    return {
        "pole": pole,
        "vec": vec,
        "n_pos": len(pos_idx), "n_neg": len(neg_idx), "n_total": len(rows),
        "score_pos": mean([scores[i] for i in range(len(rows)) if rows[i][0] == "pos"]),
        "score_neg": mean([scores[i] for i in range(len(rows)) if rows[i][0] == "neg"]),
        "samples": samples,
        **({"min_coherence": min_coherence, "coherence": coherence} if min_coherence is not None else {}),
    }


def build_directions(pole_vecs: dict[str, torch.Tensor]) -> torch.Tensor:
    """由 10 个极点向量构造人格坐标系 U，形状 (n_layers, 5, d)，每个方向单位长度。"""
    dirs = []
    for k in DIMS:
        hi, lo = BIG_FIVE[k]
        d = pole_vecs[hi] - pole_vecs[lo]
        dirs.append(d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return torch.stack(dirs, dim=1)


def save_directions(path, U: torch.Tensor, s: torch.Tensor | None = None, meta: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"U": U, "s": s, "dims": DIMS, "meta": meta or {}}, path)


def load_directions(path) -> dict:
    return torch.load(path, map_location="cpu")


def steer_vector(U_layer: torch.Tensor, s_layer: float, z: torch.Tensor) -> torch.Tensor:
    """人格坐标 z（5 维，单位是 s_ℓ 的比例）→ 要加到第 ℓ 层残差流上的向量 (d,)。

        vec = s_ℓ · Σ_k z_k · u_k
    """
    return s_layer * (z.to(U_layer.dtype) @ U_layer)
