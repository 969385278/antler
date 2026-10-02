"""读取 configs/*.yaml，并准备输出目录。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = ROOT / cfg.get("out_dir", "outputs/default")
    out.mkdir(parents=True, exist_ok=True)
    cfg["out_path"] = out
    return cfg


def sweep_layers(cfg: dict, n_layers: int) -> list[int]:
    """层位扫描范围：配置里写了 layers 就用它，否则取模型深度 lo–hi 区间内每隔 step 层。"""
    if cfg.get("layers"):
        return list(cfg["layers"])
    lo, hi = cfg.get("layer_frac", [0.3, 0.6])
    step = cfg.get("layer_step", 2)
    return list(range(int(n_layers * lo), int(n_layers * hi) + 1, step))
