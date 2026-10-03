"""E0-c：层位 × 系数扫描，画出每个维度“系数 → 特质分数”的曲线，选出注入层和安全区间。

两种测量模式：
  --mode judge  生成回答 + 评委打分（默认，和论文一致，花 API 费用）
  --mode mcq    用 TRAIT 选择题直接读概率（不需要评委，便宜；需要先运行 scripts/prepare_trait.py）

输出 calibration_<mode>.csv，按维度和测量极点分别推荐层。高极效果为最大系数分减最小系数分，
低极效果反向计算；同时要求生成模式下连贯性 ≥ 阈值。
"""
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT, sweep_layers
from antler.core.model import n_layers
from antler.eval.judge import COHERENCE_PROMPT, TASK_PROMPT
from antler.eval.trait_score import mcq_high_prob, trait_score
from antler.traits.data import BIG_FIVE, DIM_NAMES, DIMS, KIT, load_jsonl
from antler.traits.vectors import load_directions, steer_vector, steer_vector_poles


def add_arguments(ap):
    ap.add_argument("--mode", choices=["judge", "mcq"], default="judge")
    ap.add_argument("--axis", choices=["bipolar", "pole"], default="bipolar",
                    help="注入使用双极轴或按符号选择的单极向量；测量极点仍由 --pole 指定")
    ap.add_argument("--pole", choices=["high", "low", "both"], default="high",
                    help="测量高极、低极或分别测两极（低极和双极仅用于 judge 模式）")
    ap.add_argument("--dims", nargs="*", default=DIMS)
    ap.add_argument("--mcq-per-dim", type=int, default=100)
    ap.add_argument("--min-coherence", type=float, default=75)
    ap.add_argument("--layers", nargs="+", type=int, help="覆盖配置中的扫描层号（从 0 开始）")
    ap.add_argument("--coefs", nargs="+", type=float, help="覆盖配置中的扫描系数")
    ap.add_argument("--n-samples", type=int, default=1, help="生成式测量每题采样次数，默认 1")
    ap.add_argument("--save-samples", action="store_true", help="逐点保存生成回答和评分（仅 judge 模式）")
    ap.add_argument("--reuse-baseline", action="store_true", help="同一次运行内，每个维度的零注入只测一次")
    ap.add_argument("--with-task", action="store_true", help="补打任务理解分（探索性评分，尚待作者盲标验证）")
    ap.add_argument("--max-new-tokens", type=int, help="覆盖评测生成长度；省略时沿用配置（不影响提取长度）")


def scan_settings(args, cfg, count):
    """命令行优先；未提供新选项时保留原有配置和测量方式。"""
    layers = args.layers if args.layers is not None else sweep_layers(cfg, count)
    coefs = args.coefs if args.coefs is not None else cfg.get("coefs", [-0.2, -0.1, 0.0, 0.1, 0.2])
    if not layers or any(l < 0 or l >= count for l in layers) or len(set(layers)) != len(layers):
        raise ValueError(f"扫描层号必须唯一且位于 0–{count - 1}")
    if not coefs or any(not math.isfinite(c) for c in coefs) or len(set(coefs)) != len(coefs):
        raise ValueError("扫描系数必须非空、有限且不重复")
    if not args.dims or any(k not in DIMS for k in args.dims) or len(set(args.dims)) != len(args.dims):
        raise ValueError("扫描维度必须是 O/C/E/A/N 中不重复的维度")
    if args.n_samples < 1:
        raise ValueError("每题采样次数必须大于 0")
    if args.mode == "mcq" and (args.save_samples or args.n_samples != 1):
        raise ValueError("--save-samples 和多次采样仅适用于 judge 模式")
    if args.mode == "mcq" and args.pole != "high":
        raise ValueError("--pole low/both 需要使用 judge 模式和对应极点的评测题")
    if args.mode == "mcq" and (args.with_task or args.max_new_tokens is not None):
        raise ValueError("--with-task 和 --max-new-tokens 仅适用于 judge 模式")
    if args.mode == "judge":
        generation_limit(args, cfg)
    return layers, coefs


def generation_limit(args, cfg):
    """生成长度只覆盖当前测量，保持配置及提取阶段不变。"""
    limit = args.max_new_tokens if args.max_new_tokens is not None else cfg.get("max_new_tokens", 200)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_new_tokens 必须是正整数")
    return limit


def measurement_poles(args):
    """双极测量分别生成两组题的回答，不把同一回答换提示重评分。"""
    return ["high", "low"] if args.pole == "both" else [args.pole]


def _redacted(value):
    """元数据保留配置，但不写入认证凭据。"""
    private = {"api_key", "judge_api_key", "hf_token", "token", "access_token", "password", "secret", "authorization"}
    if isinstance(value, dict):
        return {k: ("[已隐藏]" if k.lower() in private else _redacted(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(v) for v in value]
    return value


@contextmanager
def record_run(args, cfg, judge, layers, coefs):
    """开启逐题保存时，同时记录本次精测的来源和结束状态。"""
    if not (args.save_samples or args.with_task or args.max_new_tokens is not None or args.axis == "pole"):
        yield
        return

    def git(*argv):
        try:
            return subprocess.check_output(["git", *argv], cwd=ROOT, text=True, encoding="utf-8",
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    now = lambda: datetime.now(timezone.utc).isoformat()
    state = git("status", "--porcelain")
    files = [KIT / "trait_eval" / f"{BIG_FIVE[k][0 if pole == 'high' else 1]}.json"
             for k in args.dims for pole in measurement_poles(args)]
    files += [cfg["out_path"] / "directions.pt"]
    if args.axis == "pole":
        files += [cfg["out_path"] / "vectors" / f"{p}.pt" for k in args.dims for p in BIG_FIVE[k]]
    meta = {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(state) if state is not None else None,
            "config_path": args.config, "config": _redacted({k: v for k, v in cfg.items() if k != "out_path"}),
            "argv": sys.argv[1:], "start_time": now(), "end_time": None, "status": "running",
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "judge": _redacted({k: getattr(judge, k, None) for k in ("model", "base_url", "max_tokens", "extra_body")}),
            "seed": 0, "layers": layers, "coefs": coefs, "n_samples": args.n_samples,
            "dims": args.dims, "pole": args.pole, "poles": measurement_poles(args), "axis": args.axis,
            "max_new_tokens": generation_limit(args, cfg) if args.mode == "judge" else None,
            "with_task": args.with_task,
            "task_score_validation": "unvalidated" if args.with_task else "not_enabled",
            "task_score_use": "exploratory" if args.with_task else "not_enabled",
            "prompt_hashes": {"COHERENCE_PROMPT": hashlib.sha256(COHERENCE_PROMPT.encode("utf-8")).hexdigest(),
                              **({"TASK_PROMPT": hashlib.sha256(TASK_PROMPT.encode("utf-8")).hexdigest()}
                                 if args.with_task else {})},
            "reuse_baseline": args.reuse_baseline,
            "data_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}}
    path = cfg["out_path"] / "run_meta.json"

    def save():
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    save()
    try:
        yield
    except BaseException:
        meta["status"] = "failed"
        raise
    else:
        meta["status"] = "completed"
    finally:
        meta["end_time"] = now()
        save()


def main():
    args = parse_args(add_arguments)
    cfg, model, tok, judge = setup(args, need_judge=(args.mode == "judge"))
    layers, coefs = scan_settings(args, cfg, n_layers(model))
    if args.mode == "judge" and judge is None:
        raise ValueError("judge 模式需要启用评委，请检查 use_judge 和评委环境变量")
    D = load_directions(cfg["out_path"] / "directions.pt")
    U, s = D["U"], D["s"]
    pole_vectors = {}
    if args.axis == "pole":
        for k in args.dims:
            for name in BIG_FIVE[k]:
                vector = torch.load(cfg["out_path"] / "vectors" / f"{name}.pt",
                                    map_location="cpu", weights_only=True)
                if vector.shape != (U.shape[0], U.shape[2]) or not torch.isfinite(vector).all():
                    raise ValueError(f"{name}: 单极向量与双极坐标系形状不一致或数值无效")
                pole_vectors[name] = vector
        # 在创建输出文件前验证所有将使用的非零注入，避免错误覆盖已有结果。
        for layer in layers:
            for k in args.dims:
                for coefficient in coefs:
                    coordinates = torch.zeros(len(DIMS))
                    coordinates[DIMS.index(k)] = coefficient
                    steer_vector_poles({p: v[layer] for p, v in pole_vectors.items()}, float(s[layer]), coordinates)
    poles = measurement_poles(args)
    print(f"[扫描] 层 {layers}，系数 {coefs}，维度 {args.dims}，模式 {args.mode}，测量极点 {poles}")
    if args.with_task:
        print("[任务理解] 本次分数仅用于探索性诊断；尚未完成作者盲标一致性验证。")

    mcq_items = {}
    if args.mode == "mcq":
        rows = load_jsonl("data/trait_big5_mcq.jsonl")
        rng = random.Random(0)
        for k in args.dims:
            items = [r for r in rows if r["trait"].lower() == DIM_NAMES[k].lower()]
            rng.shuffle(items)
            mcq_items[k] = items[:args.mcq_per_dim]

    out_csv = cfg["out_path"] / f"calibration_{args.mode}.csv"
    samples_path = cfg["out_path"] / f"calibration_{args.mode}_samples.jsonl"
    # 基线按维度和极点分别缓存；不同极点使用不同题目，不能互相复用。
    baselines = {}
    results = []
    sample_file = open(samples_path, "w", encoding="utf-8", newline="\n") if args.save_samples else nullcontext(None)
    with record_run(args, cfg, judge, layers, coefs), sample_file as sf, open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 新字段追加在末尾，保留原有五列的位置和含义。
        w.writerow(["layer", "dim", "coef", "trait", "coherence", "pole"] + (["task"] if args.with_task else []))
        for l in layers:
            for k in args.dims:
                ki = DIMS.index(k)
                for pole in poles:
                    baseline_key = (k, pole)
                    for c in coefs:
                        z = torch.zeros(len(DIMS))
                        z[ki] = c
                        if c == 0:
                            steer = None
                        elif args.axis == "pole":
                            steer = {l: steer_vector_poles({p: v[l] for p, v in pole_vectors.items()}, float(s[l]), z)}
                        else:
                            steer = {l: steer_vector(U[l], float(s[l]), z)}
                        reused = args.reuse_baseline and c == 0 and baseline_key in baselines
                        if reused:
                            r, source_layer = baselines[baseline_key]
                            trait, coh = r["trait"], r.get("coherence")
                        elif args.mode == "judge":
                            r = trait_score(model, tok, judge, k, steer=steer, pole=pole,
                                            n_questions=cfg.get("calib_questions", 10),
                                            max_new_tokens=generation_limit(args, cfg), n_samples=args.n_samples,
                                            with_task=args.with_task)
                            trait, coh = r["trait"], r.get("coherence")
                        else:
                            probs = mcq_high_prob(model, tok, mcq_items[k], steer=steer)
                            trait, coh = 100 * sum(probs) / len(probs), None
                            r = {"trait": trait, "coherence": coh}
                        if not reused:
                            source_layer = l
                            if args.reuse_baseline and c == 0:
                                baselines[baseline_key] = (r, l)
                        if sf is not None:
                            # 复用的基线仍写完整记录，并标明来源，不能当作独立重复样本。
                            record = {"layer": l, "dim": k, "coef": c, "pole": pole, "axis": args.axis,
                                      "trait_name": BIG_FIVE[k][0 if pole == "high" else 1],
                                      "n_samples": args.n_samples, "baseline_reused": reused,
                                      "max_new_tokens": generation_limit(args, cfg), "with_task": args.with_task,
                                      "source_layer": source_layer, "result": r}
                            sf.write(json.dumps(record, ensure_ascii=False) + "\n")
                            sf.flush()
                        w.writerow([l, k, c, trait, coh, pole] + ([r.get("task")] if args.with_task else []))
                        f.flush()
                        results.append((l, k, c, trait, coh, pole))
                        task_log = f" task={r.get('task')}" if args.with_task else ""
                        print(f"  layer {l:2d} {k} {pole} coef {c:+.2f}: trait={trait} coherence={coh}{task_log}")

    print("\n每个维度与测量极点的推荐注入层（效果最大且连贯）：")
    cmax, cmin = max(coefs), min(coefs)
    for k in args.dims:
        for pole in poles:
            best = None
            for l in layers:
                group = [r for r in results if r[0] == l and r[1] == k and r[5] == pole]
                hi = [r for r in group if r[2] == cmax]
                lo = [r for r in group if r[2] == cmin]
                if not hi or not lo or hi[0][3] is None or lo[0][3] is None:
                    continue
                cohs = [r[4] for r in group if r[4] is not None]
                if cohs and min(cohs) < args.min_coherence:
                    continue
                eff = hi[0][3] - lo[0][3]
                if pole == "low":
                    eff = -eff
                if best is None or eff > best[1]:
                    best = (l, eff)
            print(f"  {k} {pole}: " + (f"layer {best[0]}，效果 {best[1]:.1f}" if best else "没有满足连贯性要求的层，缩小系数后重试"))
    print(f"\n完整结果：{out_csv}")
    if args.save_samples:
        print(f"逐点回答与评分：{samples_path}")


if __name__ == "__main__":
    main()
