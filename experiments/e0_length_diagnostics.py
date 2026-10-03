"""C20：在同一条原始 token 流上比较 320 前缀与长回答，仅作 calib 长度诊断。"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model
from antler.eval.judge import Judge, COHERENCE_PROMPT, TASK_PROMPT
from antler.traits.data import DIMS, KIT, load_trait
from antler.traits.vectors import steer_vector
from e0_diagnostics import now, read_json, write_json, sha256
from e0_extract_coherence import _redacted
from e0_refilter import _git

CASES = (("C", "careless", -0.2), ("N", "nervous", 0.2))


def add_arguments(ap):
    ap.add_argument("--refilter-run", help="已完成的三组对照目录，默认取配置目录下 latest.json")
    ap.add_argument("--long-tokens", type=int, default=1024, help="独立长度诊断的上限，默认1024")
    ap.add_argument("--layer", type=int, default=16)


def token_views(item, tokenizer, eos_ids, long_limit):
    """验证 token 与停止信息一致，再直接切原始 token；不对文本重新分词。"""
    ids = item["token_ids"]
    if (not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids)
            or len(ids) != item["n_generated_tokens"] or len(ids) > long_limit):
        raise ValueError("原始 token 与生成计数不一致")
    first_eos = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    reason = "eos" if first_eos is not None else "length"
    if (reason != item["finish_reason"] or (first_eos is not None and first_eos != len(ids) - 1)
            or (reason == "length" and len(ids) != long_limit)):
        raise ValueError("原始 token 与停止原因不一致，暂停后续评分")
    views = {}
    for name, selected in (("prefix320", ids[:320]), ("long", ids)):
        views[name] = {"text": tokenizer.batch_decode([selected], skip_special_tokens=True)[0],
                       "n_tokens": len(selected),
                       "finish_reason": "eos" if selected[-1] in eos_ids else "length"}
    if views["long"]["text"] != item["text"]:
        raise ValueError("原始 token 解码与保存的回答不一致")
    return views


def score_views(judge, question, prompt, views):
    """相同文本只评一次；分别保留缺失分，不把缺失当作0。"""
    scored = {}
    for name, view in views.items():
        text = view["text"]
        if text not in scored:
            scored[text] = {"trait": judge.trait_scores(prompt, [question], [text])[0],
                            "coherence": judge.coherence_scores([question], [text])[0],
                            "task": judge.task_scores([question], [text])[0]}
        view["scores"] = dict(scored[text])


def report(output, rows, status):
    fields = ["variant", "dim", "question_index", "coef", "view", "n_tokens", "finish_reason",
              "trait", "coherence", "task"]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            for name, view in row["views"].items():
                writer.writerow({**{k: row[k] for k in fields[:4]}, "view": name,
                                 **{k: view[k] for k in ("n_tokens", "finish_reason")}, **view.get("scores", {})})
    lines = ["# C20 小批长度诊断", "", f"状态：{status}；生成 {len(rows)} 条回答。",
             "仅使用两个极点各前两道 calib 题，每条件采样一次；不作正式验收。",
             "前320与长版本来自同一条token流；新前缀不等于此前810条里的回答。任务评分尚未人工验证。",
             "long若仍为length，表示达到本轮上限，并非自然完整回答。", "",
             "| 组 | 维度/题号 | 长度/结束 | Δ人格 | Δ连贯 | Δ理解 |", "|---|---|---|---|---|---|"]
    for row in rows:
        short, long = row["views"]["prefix320"], row["views"]["long"]
        changes = []
        for key in ("trait", "coherence", "task"):
            a, b = short.get("scores", {}).get(key), long.get("scores", {}).get(key)
            changes.append("缺失" if a is None or b is None else f"{b-a:+.1f}")
        lines.append(f"| {row['variant']} | {row['dim']}/{row['question_index']} | "
                     f"{long['n_tokens']}/{long['finish_reason']} | " + " | ".join(changes) + " |")
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    with zipfile.ZipFile(output / "results.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("run_meta.json", "generation_settings.json", "samples.jsonl", "summary.csv", "summary.md"):
            bundle.write(output / name, name)


def run_diagnostic(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    if args.long_tokens <= 320:
        raise ValueError("长版本上限必须大于320")
    source = Path(cfg["out_path"]).resolve()
    parent = Path(args.refilter_run).resolve() if args.refilter_run else (
        source / "refilter_diagnostics" / read_json(source / "refilter_diagnostics" / "latest.json")["run_id"])
    parent_meta = read_json(parent / "run_meta.json")
    if parent_meta.get("status") != "completed":
        raise ValueError("三组对照尚未完成")
    paths = {v: parent / "variants" / v / "directions.pt" for v in ("original", "cohfilt")}
    expected = {"original": parent_meta["provenance"]["file_hashes"][str(source / "directions.pt")],
                "cohfilt": parent_meta["steps"]["refilter_cohfilt"]["artifact_hashes"]["variants/cohfilt/directions.pt"]}
    directions = {}
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"{name} 方向与上轮对照哈希不一致")
        directions[name] = torch.load(path, map_location="cpu", weights_only=True)
        d = directions[name]
        if (d["dims"] != DIMS or not 0 <= args.layer < d["U"].shape[0]
                or not torch.isfinite(d["U"]).all() or not torch.isfinite(d["s"]).all()):
            raise ValueError("方向、层号或尺度无效")
    if (directions["original"]["U"].shape != directions["cohfilt"]["U"].shape
            or not torch.equal(directions["original"]["s"], directions["cohfilt"]["s"])):
        raise ValueError("两组方向的形状或尺度不一致")
    traits = {pole: load_trait(pole, "eval") for _, pole, _ in CASES}
    files = [Path(args.config).resolve(), parent / "run_meta.json", *paths.values()]
    files += [KIT / "trait_eval" / f"{pole}.json" for _, pole, _ in CASES]
    files += list((ROOT / "antler").rglob("*.py")) + [Path(__file__).resolve()]
    model_dir = ROOT / cfg["model"]
    files += sorted(model_dir.glob("*.json"))
    out = source / "length_diagnostics" / (now().replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    # 缓存只复制到本轮；原对照的结果与缓存保持不变。
    if (parent / "judge_cache.jsonl").exists():
        shutil.copy2(parent / "judge_cache.jsonl", out / "judge_cache.jsonl")
    meta = {"task": "C20", "experiment": "e0_length_diagnostics", "status": "running",
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "config_path": str(Path(args.config).resolve()), "config": _redacted(cfg), "argv": sys.argv[1:],
            "start_time": now(), "end_time": None, "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": None, "judge": None, "seed": 0, "batch_size": 1,
            "layer": args.layer, "long_tokens": args.long_tokens, "prefix_tokens": 320,
            "question_indices": [0, 1], "task_score_validation": "unvalidated", "scope": "calib_exploratory",
            "prompt_hashes": {"COHERENCE_PROMPT": hashlib.sha256(COHERENCE_PROMPT.encode()).hexdigest(),
                              "TASK_PROMPT": hashlib.sha256(TASK_PROMPT.encode()).hexdigest()},
            "data_hashes": {str(p): sha256(p) for p in files}, "refilter_run": str(parent)}
    write_json(out / "run_meta.json", meta)
    rows = []
    def save_rows():
        (out / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in rows),
                                          encoding="utf-8", newline="\n")
    try:
        model, tok = load_model(cfg["model"], cfg.get("dtype", "bfloat16"), cfg.get("device", "auto"))
        shape = directions["original"]["U"].shape
        if (model.config.num_hidden_layers, model.config.hidden_size) != (shape[0], shape[2]):
            raise ValueError("模型与方向形状不一致")
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        if not eos:
            raise ValueError("实际生成配置缺少EOS，停止诊断")
        settings = {"model_generation_config": model.generation_config.to_dict(),
                    "model_eos": model.config.eos_token_id, "tokenizer_eos": tok.eos_token_id,
                    "tokenizer_pad": tok.pad_token_id, "effective_eos": eos,
                    "call_overrides": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": args.long_tokens,
                                       "do_sample": True, "pad_token_id": tok.pad_token_id}}
        write_json(out / "generation_settings.json", settings)
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        for variant in ("baseline", "original", "cohfilt"):
            for dim, pole, coefficient in CASES:
                for qi in (0, 1):
                    question = traits[pole]["questions"][qi]
                    coef = 0 if variant == "baseline" else coefficient
                    steer = None
                    if coef:
                        d = directions[variant]
                        z = torch.zeros(5); z[DIMS.index(dim)] = coef
                        steer = {args.layer: steer_vector(d["U"][args.layer], float(d["s"][args.layer]), z)}
                    item = generate(model, tok, [chat_prompt(tok, question)], steer=steer,
                                    max_new_tokens=args.long_tokens, temperature=1.0, top_p=1.0, batch_size=1,
                                    seed=0, return_details=True, return_token_ids=True)[0]
                    views = token_views(item, tok, eos, args.long_tokens)
                    rows.append({"variant": variant, "dim": dim, "pole": pole, "coef": coef,
                                 "question_index": qi, "question": question, "token_ids": item["token_ids"], "views": views})
                    save_rows()
                    print(f"[生成] {variant} {dim} 题{qi}：{item['n_generated_tokens']} token，{item['finish_reason']}", flush=True)
            if variant == "baseline" and not any(r["views"]["long"]["finish_reason"] == "eos" for r in rows):
                meta["status"] = "stopped_no_baseline_eos"
                print("[停止] 四条基线都未自然结束，保留原始token，不调用评委。", flush=True)
                break
        if meta["status"] == "running":
            judge = Judge(cache_path=str(out / "judge_cache.jsonl"))
            if judge.model != parent_meta["provenance"]["judge"]["model"]:
                raise ValueError("评委与上轮不一致，停止评分")
            meta["judge"] = _redacted({k: getattr(judge, k) for k in ("model", "base_url", "max_tokens", "extra_body")})
            for row in rows:
                score_views(judge, row["question"], traits[row["pole"]]["eval_prompt"], row["views"])
                save_rows()
                print(f"[评分] {row['variant']} {row['dim']} 题{row['question_index']}：{row['views']['prefix320']['scores']} → {row['views']['long']['scores']}", flush=True)
            meta["status"] = "completed"
        for name, expected_hash in meta["data_hashes"].items():
            if sha256(name) != expected_hash:
                raise ValueError("实验期间输入被更改")
    except BaseException as exc:
        meta.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        meta["end_time"] = now()
        save_rows()
        meta["artifact_hashes"] = {p.name: sha256(p) for p in (out / "samples.jsonl", out / "generation_settings.json") if p.exists()}
        write_json(out / "run_meta.json", meta)
    report(out, rows, meta["status"])
    print(f"[结果目录] {out}", flush=True)
    return out


if __name__ == "__main__":
    run_diagnostic(parse_args(add_arguments))
