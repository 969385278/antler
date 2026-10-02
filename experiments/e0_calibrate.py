"""E0-c：层位 × 系数扫描，画出每个维度“系数 → 特质分数”的曲线，选出注入层和安全区间。

两种测量模式：
  --mode judge  生成回答 + 评委打分（默认，和论文一致，花 API 费用）
  --mode mcq    用 TRAIT 选择题直接读概率（不需要评委，便宜；需要先运行 scripts/prepare_trait.py）

输出 calibration_<mode>.csv，以及每个维度推荐的层：效果 = 分数(+最大系数) − 分数(−最大系数)，
同时要求生成模式下连贯性 ≥ 阈值。
"""
import csv
import random

import torch

from _common import parse_args, setup
from antler.config import sweep_layers
from antler.core.model import n_layers
from antler.eval.trait_score import mcq_high_prob, trait_score
from antler.traits.data import DIM_NAMES, DIMS, load_jsonl
from antler.traits.vectors import load_directions, steer_vector


def main():
    def extra(ap):
        ap.add_argument("--mode", choices=["judge", "mcq"], default="judge")
        ap.add_argument("--dims", nargs="*", default=DIMS)
        ap.add_argument("--mcq-per-dim", type=int, default=100)
        ap.add_argument("--min-coherence", type=float, default=75)
    args = parse_args(extra)
    cfg, model, tok, judge = setup(args, need_judge=(args.mode == "judge"))
    D = load_directions(cfg["out_path"] / "directions.pt")
    U, s = D["U"], D["s"]
    layers = sweep_layers(cfg, n_layers(model))
    coefs = cfg.get("coefs", [-0.2, -0.1, 0.0, 0.1, 0.2])
    print(f"[扫描] 层 {layers}，系数 {coefs}，维度 {args.dims}，模式 {args.mode}")

    mcq_items = {}
    if args.mode == "mcq":
        rows = load_jsonl("data/trait_big5_mcq.jsonl")
        rng = random.Random(0)
        for k in args.dims:
            items = [r for r in rows if r["trait"].lower() == DIM_NAMES[k].lower()]
            rng.shuffle(items)
            mcq_items[k] = items[:args.mcq_per_dim]

    out_csv = cfg["out_path"] / f"calibration_{args.mode}.csv"
    results = []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "dim", "coef", "trait", "coherence"])
        for l in layers:
            for i, k in enumerate(args.dims):
                ki = DIMS.index(k)
                for c in coefs:
                    z = torch.zeros(len(DIMS))
                    z[ki] = c
                    steer = None if c == 0 else {l: steer_vector(U[l], float(s[l]), z)}
                    if args.mode == "judge":
                        r = trait_score(model, tok, judge, k, steer=steer, n_questions=cfg.get("calib_questions", 10),
                                        max_new_tokens=cfg.get("max_new_tokens", 200))
                        trait, coh = r["trait"], r.get("coherence")
                    else:
                        probs = mcq_high_prob(model, tok, mcq_items[k], steer=steer)
                        trait, coh = 100 * sum(probs) / len(probs), None
                    w.writerow([l, k, c, trait, coh])
                    f.flush()
                    results.append((l, k, c, trait, coh))
                    print(f"  layer {l:2d} {k} coef {c:+.2f}: trait={trait} coherence={coh}")

    print("\n每个维度的推荐注入层（效果最大且连贯）：")
    cmax, cmin = max(coefs), min(coefs)
    for k in args.dims:
        best = None
        for l in layers:
            hi = [r for r in results if r[0] == l and r[1] == k and r[2] == cmax]
            lo = [r for r in results if r[0] == l and r[1] == k and r[2] == cmin]
            if not hi or not lo or hi[0][3] is None or lo[0][3] is None:
                continue
            cohs = [r[4] for r in results if r[0] == l and r[1] == k and r[4] is not None]
            if cohs and min(cohs) < args.min_coherence:
                continue
            eff = hi[0][3] - lo[0][3]
            if best is None or eff > best[1]:
                best = (l, eff)
        print(f"  {k}: " + (f"layer {best[0]}，效果 {best[1]:.1f}" if best else "没有满足连贯性要求的层，缩小系数后重试"))
    print(f"\n完整结果：{out_csv}")


if __name__ == "__main__":
    main()
