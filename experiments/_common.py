"""实验脚本的公共部分：读配置、加载模型和评委。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from antler.config import load_config  # noqa: E402
from antler.core.model import load_model  # noqa: E402


def parse_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.yaml")
    if extra:
        extra(ap)
    return ap.parse_args()


def setup(args, need_judge: bool = True):
    cfg = load_config(args.config)
    print(f"[配置] {args.config} → 输出到 {cfg['out_path']}")
    model, tok = load_model(cfg["model"], cfg.get("dtype", "bfloat16"), cfg.get("device", "auto"))
    judge = None
    if need_judge and cfg.get("use_judge", True):
        from antler.eval.judge import Judge
        judge = Judge(cache_path=str(cfg["out_path"] / "judge_cache.jsonl"))
    return cfg, model, tok, judge


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
