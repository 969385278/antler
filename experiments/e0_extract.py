"""E0-b：提取大五 10 个极点的人格向量，构造人格坐标系 U（每层 5 个方向）。

需要先运行 e0_layer_norms.py。输出：
  vectors/<极点>.pt          每层一个向量
  vectors/<极点>_samples.json 生成的回答和评分（用于人工检查）
  directions.pt               {"U": (L,5,d), "s": (L,), "dims": [...]}
"""
import torch

from _common import dump, parse_args, setup
from antler.traits.data import BIG_FIVE, DIMS, POLES, check_no_overlap
from antler.traits.vectors import build_directions, extract_pole, save_directions


def main():
    args = parse_args(lambda ap: ap.add_argument("--poles", nargs="*", default=None))
    cfg, model, tok, judge = setup(args)
    assert all(v == 0 for v in check_no_overlap().values()), "提取集和评测集有重叠，停止"
    s = torch.load(cfg["out_path"] / "layer_norms.pt")
    vdir = cfg["out_path"] / "vectors"
    vdir.mkdir(exist_ok=True)

    for pole in args.poles or POLES:
        path = vdir / f"{pole}.pt"
        if path.exists():
            print(f"[跳过] {pole} 已存在")
            continue
        r = extract_pole(model, tok, pole, judge=judge, n_rollouts=cfg.get("n_rollouts", 2),
                         max_new_tokens=cfg.get("max_new_tokens", 200), batch_size=cfg.get("batch_size", 16),
                         max_questions=cfg.get("max_questions"))
        torch.save(r["vec"], path)
        dump(r["samples"], vdir / f"{pole}_samples.json")
        print(f"[{pole}] 保留 pos={r['n_pos']} neg={r['n_neg']} / {r['n_total']}，"
              f"平均特质分 pos={r['score_pos']} neg={r['score_neg']}")

    if not all((vdir / f"{p}.pt").exists() for p in POLES):
        print("还有极点没提取完，暂不构造坐标系")
        return
    pole_vecs = {p: torch.load(vdir / f"{p}.pt") for p in POLES}
    U = build_directions(pole_vecs)
    save_directions(cfg["out_path"] / "directions.pt", U, s, meta={"model": cfg["model"]})

    # 两个诊断：向量的天然大小（用于设定校准系数的范围）、五个方向之间是否近似正交
    print("\n各层“极点向量长度 / s_ℓ”（天然的推动幅度，校准系数可以参考这个量级）：")
    for l in range(U.shape[0]):
        ratios = [float((pole_vecs[BIG_FIVE[k][0]][l] - pole_vecs[BIG_FIVE[k][1]][l]).norm() / s[l]) for k in DIMS]
        print(f"  layer {l:2d}: " + "  ".join(f"{k}={r:.3f}" for k, r in zip(DIMS, ratios)))
    mid = U.shape[0] // 2
    cos = U[mid] @ U[mid].T
    print(f"\n第 {mid} 层五个方向的余弦相似度（对角线外越接近 0 越正交）：")
    print("     " + "  ".join(f"{k:>6}" for k in DIMS))
    for i, k in enumerate(DIMS):
        print(f"  {k}  " + "  ".join(f"{float(cos[i, j]):6.2f}" for j in range(len(DIMS))))


if __name__ == "__main__":
    main()
