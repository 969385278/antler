"""人格特质数据：大五的 10 个极点、提取集和评测集。

数据来自 PERSONA 仓库（data/kit/trait_extract 与 data/kit/trait_eval），两者题目互不重叠，
系统提示相同。每个文件：{"instruction": [{"pos":..., "neg":...} ×5], "questions": [...×20], "eval_prompt": "..."}
"""
from __future__ import annotations

import json
from pathlib import Path

KIT = Path(__file__).resolve().parents[2] / "data" / "kit"

# 大五维度 → (高极, 低极)
BIG_FIVE = {
    "O": ("inventive", "consistent"),
    "C": ("dependable", "careless"),
    "E": ("outgoing", "solitary"),
    "A": ("compassionate", "self-interested"),
    "N": ("nervous", "calm"),
}
DIMS = list(BIG_FIVE)                       # ["O","C","E","A","N"]，人格坐标的顺序
POLES = [p for pair in BIG_FIVE.values() for p in pair]
DIM_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
             "A": "Agreeableness", "N": "Neuroticism"}


def load_trait(pole: str, split: str = "extract") -> dict:
    """split: "extract"（提取向量、做探针）或 "eval"（测人格分数）。两者绝不能混用。"""
    folder = {"extract": "trait_extract", "eval": "trait_eval"}[split]
    with open(KIT / folder / f"{pole}.json", encoding="utf-8") as f:
        return json.load(f)


def check_no_overlap() -> dict[str, int]:
    """确认提取集和评测集没有重复题目，返回每个极点的重叠数（应全为 0）。"""
    return {p: len(set(load_trait(p, "extract")["questions"]) & set(load_trait(p, "eval")["questions"]))
            for p in POLES}


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_personality_bench(dim: str) -> list[str]:
    """PersonalityBench（NPTI）的情境题，dim 为 "O"/"C"/"E"/"A"/"N"。文件是 JSONL 格式。"""
    rows = load_jsonl(KIT / "personality_bench" / f"{DIM_NAMES[dim]}.json")
    return [r["question"] for r in rows]
