"""E1：单段经历能否写成人格位移 δ？

对 data/pilot_experiences.jsonl 中的每段经历：
  - 方法 A（投影）得到 δ_A；
  - 方法 B（蒸馏，--distill）以 δ_A 初始化，得到 δ_B，并报告蒸馏前后的 KL；
  - 检查 δ 的符号是否和标注的预期影响一致（方向准确率）。
加 --behavior 时，再用评委测三种条件下各维度的特质分：基线 / 经历放在上下文 / 只注入 δ（花 API 费用）。
"""
import torch

from _common import dump, parse_args, setup
from antler.traits.data import DIMS, load_jsonl
from antler.traits.vectors import load_directions
from antler.memory.write import delta_distill, delta_projection, kl_without_steer, probe_questions


def sign_agreement(delta, intended):
    hits = [(1 if float(delta[DIMS.index(k)]) * v > 0 else 0) for k, v in intended.items()]
    return sum(hits), len(hits)


def main():
    def extra(ap):
        ap.add_argument("--layer", type=int, required=True, help="E0 校准选出的注入层")
        ap.add_argument("--distill", action="store_true")
        ap.add_argument("--behavior", action="store_true")
        ap.add_argument("--experiences", default="data/pilot_experiences.jsonl")
        ap.add_argument("--baseline", choices=["neutral", "none"], default="neutral",
                        help="对照组：neutral=中性记忆（推荐），none=无记忆（会混入格式效应）")
    args = parse_args(extra)
    cfg, model, tok, judge = setup(args, need_judge=args.behavior)
    D = load_directions(cfg["out_path"] / "directions.pt")
    U_l, s_l = D["U"][args.layer], float(D["s"][args.layer])
    probes = probe_questions(cfg.get("probe_n", 30))
    eps = load_jsonl(args.experiences)

    report, hitsA, hitsB, tot = [], 0, 0, 0
    for ep in eps:
        dA = delta_projection(model, tok, ep["text"], probes, U_l, s_l, args.layer, baseline=args.baseline)
        h, n = sign_agreement(dA, ep["intended"])
        hitsA += h
        tot += n
        row = {"id": ep["id"], "intended": ep["intended"], "delta_A": dA.tolist()}
        if args.distill:
            kl0 = kl_without_steer(model, tok, ep["text"], probes, baseline=args.baseline)
            dB, log = delta_distill(model, tok, ep["text"], probes, U_l, s_l, args.layer, init=dA,
                                    baseline=args.baseline)
            hitsB += sign_agreement(dB, ep["intended"])[0]
            explained = (kl0 - log[-1]["kl"]) / kl0 if kl0 > 0 else None  # 人格解释率
            row.update({"delta_B": dB.tolist(), "kl_no_steer": kl0, "kl_final": log[-1]["kl"],
                        "explained": explained, "log": log})
        if args.behavior:
            from antler.eval.trait_score import trait_score
            from antler.memory.write import BASE_SYSTEM, MEMORY_TEMPLATE, NEUTRAL_MEMORY
            from antler.traits.vectors import steer_vector
            d_use = row.get("delta_B", row["delta_A"])
            steer = {args.layer: steer_vector(U_l, s_l, torch.tensor(d_use))}
            mem = MEMORY_TEMPLATE.format(experience=ep["text"])
            row["behavior"] = {}
            for k in ep["intended"]:
                base_sys = MEMORY_TEMPLATE.format(experience=NEUTRAL_MEMORY) if args.baseline == "neutral" else BASE_SYSTEM  # 行为测量用第 1 条中性记忆
                base = trait_score(model, tok, judge, k, n_questions=10, with_coherence=False, system=base_sys)
                ctx = trait_score(model, tok, judge, k, n_questions=10, with_coherence=False, system=mem)
                inj = trait_score(model, tok, judge, k, steer=steer, n_questions=10, system=base_sys)
                row["behavior"][k] = {"baseline": base["trait"], "in_context": ctx["trait"],
                                      "injected": inj["trait"], "coherence": inj["coherence"]}
                print(f"    {k}: 基线 {base['trait']}  上下文 {ctx['trait']}  注入 {inj['trait']}")
        report.append(row)
        fmt = lambda d: " ".join(f"{k}={x:+.3f}" for k, x in zip(DIMS, d))
        print(f"[{ep['id']}] 预期 {ep['intended']}\n    δ_A: {fmt(row['delta_A'])}"
              + (f"\n    δ_B: {fmt(row['delta_B'])}  KL {row['kl_no_steer']:.4f} → {row['kl_final']:.4f}"
               f"  人格解释率 {100 * row['explained']:.1f}%" if args.distill else ""))

    # 共模诊断：所有经历 δ_A 的平均值，以及去掉平均后各经历自己的偏移有多大
    allA = torch.tensor([r["delta_A"] for r in report])
    common = allA.mean(0)
    spread = (allA - common).norm(dim=1).mean()
    print("\n共模偏移（所有经历 δ_A 的平均）：", " ".join(f"{k}={x:+.3f}" for k, x in zip(DIMS, common.tolist())))
    ratio = float(common.norm()) / max(float(spread), 1e-12)
    verdict = ("共模远大于各经历的偏离：测到的主要是格式效应，请检查对照组" if ratio > 2 else
               "共模与各经历偏离量级相当：格式效应已基本消除")
    print(f"各经历相对共模的平均偏离：{float(spread):.4f}；共模大小：{float(common.norm()):.4f}；"
          f"比值 {ratio:.1f}（{verdict}）")
    # 去共模后的方向准确率：δ 减去所有经历的平均再判断方向（要求经历集合在各方向上大致平衡）
    hc = sum(sign_agreement(torch.tensor(r["delta_A"]) - common, r["intended"])[0] for r in report)
    summary = {"layer": args.layer, "baseline": args.baseline, "sign_acc_A": hitsA / tot,
               "sign_acc_A_centered": hc / tot, "common_mode_A": common.tolist(), "spread_A": float(spread)}
    if args.distill:
        allB = torch.tensor([r["delta_B"] for r in report])
        hcb = sum(sign_agreement(torch.tensor(r["delta_B"]) - allB.mean(0), r["intended"])[0] for r in report)
        summary["sign_acc_B_centered"] = hcb / tot
    if args.distill:
        summary["sign_acc_B"] = hitsB / tot
        ex = [r["explained"] for r in report if r.get("explained") is not None]
        summary["explained_mean"] = sum(ex) / len(ex) if ex else None
    print("\n方向准确率：", {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in summary.items() if not k.startswith("common_mode")})
    dump({"summary": summary, "episodes": report}, cfg["out_path"] / f"e1_layer{args.layer}.json")


if __name__ == "__main__":
    main()
