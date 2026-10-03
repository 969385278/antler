"""人格测量。

1. trait_score：生成式测量。让模型回答某极点的评测题，由评委打 0–100 的特质分（Persona Vectors / PERSONA 的做法）。
2. mcq_high_prob：选择题测量（TRAIT 格式）。不生成、不需要评委：直接读模型在 A/B/C/D 上的概率，
   特质分 = 高特质选项的概率之和。便宜、确定，适合大规模扫描。
"""
from __future__ import annotations

import string

import torch

from ..core.generate import generate
from ..core.hooks import Steer, no_steer
from ..core.model import chat_prompt, model_device
from ..traits.data import BIG_FIVE, load_trait


def trait_score(model, tok, judge, dim: str, *, steer=None, n_questions: int = 20, n_samples: int = 1,
                max_new_tokens: int = 200, temperature: float = 1.0, seed: int = 0,
                with_coherence: bool = True, system: str | None = None, pole: str = "high",
                with_task: bool = False, split: str = "all") -> dict:
    """使用指定极点自己的评测题和提示；分数越高越体现该极点，不翻转注入方向。"""
    if pole not in ("high", "low"):
        raise ValueError("pole 必须是 high 或 low")
    if split not in ("calib", "test", "all"):
        raise ValueError("split 必须是 calib、test 或 all")
    if n_questions < 1 or n_samples < 1:
        raise ValueError("题数与采样次数必须为正整数")
    trait_name = BIG_FIVE[dim][0 if pole == "high" else 1]
    data = load_trait(trait_name, "eval")
    indices = list(range(len(data["questions"])))
    if split != "all":
        if len(indices) != 20:
            raise ValueError("固定切分要求每个极点恰好20道评测题")
        indices = indices[:10] if split == "calib" else indices[10:20]
    indices = indices[:n_questions]
    qs = [data["questions"][i] for i in indices] * n_samples
    prompts = [chat_prompt(tok, q, system=system) for q in qs]
    generated = generate(model, tok, prompts, steer=steer, max_new_tokens=max_new_tokens,
                         temperature=temperature, seed=seed, return_details=True)
    answers = [item["text"] for item in generated]
    ts = judge.trait_scores(data["eval_prompt"], qs, answers)
    out = {"dim": dim, "pole": pole, "trait_name": trait_name, "split": split,
           "trait": _mean(ts), "n_valid": sum(t is not None for t in ts)}
    samples = [{"q": q, "a": a, "trait": t, "question_index": indices[i % len(indices)],
                "sample_index": i // len(indices), "n_generated_tokens": item["n_generated_tokens"],
                "finish_reason": item["finish_reason"]}
               for i, (q, a, t, item) in enumerate(zip(qs, answers, ts, generated))]
    if with_coherence:
        cs = judge.coherence_scores(qs, answers)
        out["coherence"] = _mean(cs)
        out["n_valid_coherence"] = sum(c is not None for c in cs)
        for sample, coherence in zip(samples, cs):
            sample["coherence"] = coherence
    if with_task:
        tasks = judge.task_scores(qs, answers)
        out["task"] = _mean(tasks)
        out["n_valid_task"] = sum(t is not None for t in tasks)
        for sample, task in zip(samples, tasks):
            sample["task"] = task
    out["samples"] = samples
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


MCQ_TEMPLATE = """{question}

{options}

Choose the option that best describes what you would do. Answer with a single letter."""


def mcq_prompt(tok, question: str, options: list[str]) -> str:
    letters = string.ascii_uppercase[:len(options)]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    return chat_prompt(tok, MCQ_TEMPLATE.format(question=question, options=opts))


@torch.no_grad()
def mcq_high_prob(model, tok, items: list[dict], *, steer=None, batch_size: int = 16) -> list[float]:
    """items: [{"question", "options": [...], "is_high": [0/1, ...]}]。

    返回每题“高特质选项”的概率之和（只在选项字母上做 softmax）。
    """
    device = model_device(model)
    n_opt = len(items[0]["options"])
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in string.ascii_uppercase[:n_opt]]
    tok.padding_side = "left"
    res = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [mcq_prompt(tok, it["question"], it["options"]) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with (Steer(model, steer) if steer else no_steer()):
            logits = model(**enc).logits[:, -1, letter_ids].float()
        probs = logits.softmax(-1).cpu()
        for p, it in zip(probs, batch):
            res.append(float(sum(p[j] for j, h in enumerate(it["is_high"]) if h)))
    return res
