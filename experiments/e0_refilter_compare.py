"""一次完成 C18 的三组校准对照，使用 C20 的逐条质量与截断诊断。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from _common import parse_args, setup
from _refilter_report import finish_report
from antler.traits.data import BIG_FIVE, KIT, load_trait
from e0_diagnostics import (build_plan as base_plan, now, output_lock, read_json,
                            run_step, sha256, write_json)
from e0_extract_coherence import DEFAULT_POLES, _redacted
from e0_refilter import prepare_inputs

VARIANTS = ("original", "paired", "cohfilt")
MEASUREMENTS = {
    "c_high": ("C", "high", [0.0, 0.2]),
    "c_low": ("C", "low", [-0.2, 0.0]),
    "n_high": ("N", "high", [0.0, 0.2, 0.25]),
    "n_low": ("N", "low", [-0.2, 0.0]),
}
EVAL_FILES = ("calibration_judge.csv", "calibration_judge_samples.jsonl", "run_meta.json")
REFILTER_FILES = ("refilter_summary.json", "refilter_samples.jsonl", "run_meta.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def add_arguments(ap):
    ap.add_argument("--fresh", action="store_true", help="新建一轮；默认校验并恢复相同输入的最近一轮")


def build_plan(args, cfg):
    """全部检查通过后才启动模型或评分，保留原提取配置。"""
    prepared = [prepare_inputs(cfg, cfg["out_path"], DEFAULT_POLES, 50, paired)
                for paired in (True, False)]
    provenance = base_plan(args, cfg)["provenance"]
    provenance.pop("commands", None)
    for item in prepared:
        provenance["file_hashes"].update(item["data_hashes"])
    for dim, pole, _ in MEASUREMENTS.values():
        name = BIG_FIVE[dim][0 if pole == "high" else 1]
        path = KIT / "trait_eval" / f"{name}.json"
        data = read_json(path)
        if len(data["questions"]) < 10 or not data["eval_prompt"]:
            raise ValueError(f"{name} 缺少 calib 题或评分提示")
        provenance["file_hashes"][str(path.resolve())] = sha256(path)
    provenance.update(scope="C18 三组对照；C20 探索性质量评分；仅 calib 前10题",
                      variants=VARIANTS, measurements=MEASUREMENTS,
                      n_samples=3, max_new_tokens=320,
                      task_score_validation="unvalidated", task_score_use="exploratory",
                      expected_generated_answers=810,
                      selection_counts={name: {p: src["counts"] for p, src in item["inputs"].items()}
                                        for name, item in zip(("paired", "cohfilt"), prepared)})
    return {"fingerprint": digest(provenance), "provenance": provenance}


def build_commands(config, run_dir):
    """只给校准传 320；重算沿用原提取配置和旧回答。"""
    run_dir, config = Path(run_dir), Path(config).resolve()
    common = [sys.executable, "-u"]
    source = read_json(run_dir / "run_meta.json")["source_out"]
    commands = {}
    for variant in VARIANTS[1:]:
        commands[f"refilter_{variant}"] = common + [
            "experiments/e0_refilter.py", "--config", str(config),
            "--source-out", source, "--output-dir", str(run_dir / "variants" / variant),
            "--poles", *DEFAULT_POLES, "--min-coherence", "50",
        ] + (["--paired-only"] if variant == "paired" else [])
    for variant in VARIANTS:
        for measurement, (dim, pole, coefs) in MEASUREMENTS.items():
            commands[f"{variant}_{measurement}"] = common + [
                "experiments/e0_calibrate.py", "--config", str(run_dir / "configs" / f"{variant}.yaml"),
                "--mode", "judge", "--layers", "16", "--dims", dim, "--pole", pole,
                "--coefs", *map(str, coefs), "--n-samples", "3", "--save-samples",
                "--reuse-baseline", "--with-task", "--max-new-tokens", "320",
            ]
    return commands


def _score(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
        raise RuntimeError("评分必须为空或 0–100 的有限数值")
    return float(value)


def validate_evaluation(output, measurement):
    """核对参数、逐题数量、评分分母和CSV，拒绝不完整或陈旧输出。"""
    dim, pole, coefs = MEASUREMENTS[measurement]
    meta = read_json(output / "run_meta.json")
    wanted_meta = {"status": "completed", "layers": [16], "dims": [dim], "pole": pole,
                   "coefs": coefs, "n_samples": 3, "with_task": True, "max_new_tokens": 320,
                   "task_score_validation": "unvalidated", "task_score_use": "exploratory"}
    if not meta.get("end_time") or any(meta.get(k) != v for k, v in wanted_meta.items()):
        raise RuntimeError("校准元数据与本轮参数不符")
    if meta.get("data_hashes", {}).get(str(output / "directions.pt")) != sha256(output / "directions.pt"):
        raise RuntimeError("校准方向哈希不符")
    records = [json.loads(line) for line in (output / EVAL_FILES[1]).read_text(encoding="utf-8").splitlines()]
    with (output / EVAL_FILES[0]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = lambda items: [(int(r["layer"]), r["dim"], r["pole"], float(r["coef"])) for r in items]
    wanted = [(16, dim, pole, c) for c in coefs]
    if keys(records) != wanted or keys(rows) != wanted:
        raise RuntimeError("校准点数、顺序或参数不符")
    trait_name = BIG_FIVE[dim][0 if pole == "high" else 1]
    questions = Counter(load_trait(trait_name, "eval")["questions"][:10] * 3)
    for record, row in zip(records, rows):
        result = record["result"]
        samples = result["samples"]
        if (len(samples) != 30 or Counter(s["q"] for s in samples) != questions
                or record.get("trait_name") != trait_name or record.get("n_samples") != 3
                or record.get("max_new_tokens") != 320 or record.get("with_task") is not True):
            raise RuntimeError("每点必须含对应极点的10道 calib 题 × 3条完整记录")
        for sample in samples:
            tokens = sample.get("n_generated_tokens")
            reason = sample.get("finish_reason")
            if (not isinstance(sample.get("a"), str) or isinstance(tokens, bool)
                    or not isinstance(tokens, int) or not 1 <= tokens <= 320
                    or reason not in ("eos", "length") or (reason == "length" and tokens != 320)):
                raise RuntimeError("生成 token 数或停止原因无效")
        for field, count in (("trait", "n_valid"), ("coherence", "n_valid_coherence"), ("task", "n_valid_task")):
            values = [_score(s[field]) for s in samples]
            valid = [v for v in values if v is not None]
            mean = sum(valid) / len(valid) if valid else None
            actual = _score(result[field])
            csv_mean = _score(float(row[field])) if row[field] else None
            if (result[count] != len(valid) or (actual is None) != (mean is None)
                    or (csv_mean is None) != (mean is None)
                    or (mean is not None and (abs(actual - mean) > 1e-6 or abs(csv_mean - mean) > 1e-6))):
                raise RuntimeError(f"{field} 的汇总与逐条有效分不一致")


def validate_refilter(output, paired_only):
    meta = read_json(output / "run_meta.json")
    if (meta.get("task") != "C18" or meta.get("status") != "completed" or not meta.get("end_time")
            or meta.get("paired_only") is not paired_only or meta.get("judge_called") is not False
            or meta.get("generation_performed") is not False):
        raise RuntimeError("C18 元数据不完整或对照条件不符")
    hashes = meta.get("artifact_hashes", {})
    if not {"directions.pt", "refilter_summary.json", "refilter_samples.jsonl", "vectors/refilter_meta.json"} <= set(hashes):
        raise RuntimeError("C18 缺少向量或来源产物")
    for name, value in hashes.items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or sha256(path) != value:
            raise RuntimeError(f"C18 产物校验失败：{name}")
    return hashes


def completed(record, run_dir, dependency):
    hashes = record.get("artifact_hashes", {})
    if record.get("status") != "completed" or record.get("dependency") != dependency or not hashes:
        return False
    for name, value in hashes.items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file() or sha256(path) != value:
            return False
    return True


def preserve(path, destination, run_dir):
    """只在本轮内部移动旧产物，保留失败尝试供追查。"""
    path, destination, root = Path(path).resolve(), Path(destination).resolve(), run_dir.resolve()
    if path == root or not path.is_relative_to(root) or not destination.is_relative_to(root):
        raise ValueError("旧产物移动必须位于本轮目录内")
    if path.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)


def snapshot(output, archive, names):
    archive.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name == "run_meta.json":
            write_json(archive / name, _redacted(read_json(output / name)))
        else:
            shutil.copy2(output / name, archive / name)


def verify_inputs(plan):
    """防止长任务中途更改代码、配置或来源后把不同输入混成一轮。"""
    for name, expected in plan["provenance"].get("file_hashes", {}).items():
        path = Path(name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"运行期间输入发生变化，请重新运行以新建一轮：{path}")


def run_comparison(args, cfg):
    source = Path(cfg["out_path"]).resolve()
    base = source / "refilter_diagnostics"
    with output_lock(base):
        plan = build_plan(args, cfg)
        pointer = base / "latest.json"
        latest = read_json(pointer) if pointer.exists() else {}
        run_id = latest.get("run_id", "")
        if (not args.fresh and latest.get("fingerprint") == plan["fingerprint"]
                and run_id and Path(run_id).name == run_id):
            run_dir = base / run_id
            state = read_json(run_dir / "run_meta.json")
            if state["fingerprint"] != plan["fingerprint"] or state["source_out"] != str(source):
                raise RuntimeError("恢复记录不一致，请使用 --fresh")
        else:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
            run_dir = base / run_id
            run_dir.mkdir()
            state = {"task": "C18+C20", "experiment": "e0_refilter_compare", **plan,
                     "source_out": str(source), "start_time": now(), "end_time": None,
                     "status": "pending", "steps": {}}
            write_json(run_dir / "run_meta.json", state)
            write_json(pointer, {"run_id": run_id, "fingerprint": plan["fingerprint"]})
            if (source / "judge_cache.jsonl").is_file():
                shutil.copy2(source / "judge_cache.jsonl", run_dir / "judge_cache.jsonl")
        print(f"[对照目录] {run_dir}\n[规模] 三组方向 × 九个测量点 × 30条回答 = 810条；均为320 token上限", flush=True)
        config_dir = run_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        for variant in VARIANTS:
            out = run_dir / "variants" / variant
            variant_cfg = {k: v for k, v in cfg.items() if k != "out_path"}
            variant_cfg["out_dir"] = str(out)
            (config_dir / f"{variant}.yaml").write_text(
                yaml.safe_dump(variant_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
        original = run_dir / "variants" / "original"
        original.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "directions.pt", original / "directions.pt")
        commands = build_commands(args.config, run_dir)
        try:
            state.update(status="running", end_time=None)
            state.pop("error_type", None)
            write_json(run_dir / "run_meta.json", state)
            for step, command in commands.items():
                verify_inputs(plan)
                refilter = step.startswith("refilter_")
                variant = step.split("_", 1)[1] if refilter else step.split("_", 1)[0]
                output = run_dir / "variants" / variant
                dependency = plan["fingerprint"] if refilter else digest({
                    "input": plan["fingerprint"], "directions": sha256(output / "directions.pt"),
                    "upstream_attempt": state["steps"].get(f"refilter_{variant}", {}).get("attempt_id"),
                })
                record = state["steps"].setdefault(step, {})
                if completed(record, run_dir, dependency):
                    print(f"[跳过] {step} 产物及依赖校验通过", flush=True)
                    continue
                attempt_id = uuid.uuid4().hex[:12]
                attempt = run_dir / "attempts" / f"{step}_{attempt_id}"
                attempt.mkdir(parents=True)
                record.update(status="running", start_time=now(), end_time=None, dependency=dependency,
                              attempt_id=attempt_id, attempts=record.get("attempts", 0) + 1, artifact_hashes={})
                write_json(run_dir / "run_meta.json", state)
                archive = run_dir / "steps" / step
                preserve(archive, attempt / "previous_archive", run_dir)
                if refilter:
                    preserve(output, attempt / "previous_variant", run_dir)
                else:
                    for name in EVAL_FILES:
                        preserve(output / name, attempt / "previous" / name, run_dir)
                    if (run_dir / "judge_cache.jsonl").exists():
                        shutil.copy2(run_dir / "judge_cache.jsonl", output / "judge_cache.jsonl")
                print(f"[运行] {step}（第{record['attempts']}次尝试）", flush=True)
                try:
                    run_step(command, attempt / "log.txt")
                finally:
                    if not refilter and (output / "judge_cache.jsonl").exists():
                        shutil.copy2(output / "judge_cache.jsonl", run_dir / "judge_cache.jsonl")
                verify_inputs(plan)
                names = REFILTER_FILES if refilter else EVAL_FILES
                live_hashes = validate_refilter(output, variant == "paired") if refilter else {}
                if not refilter:
                    validate_evaluation(output, step.split("_", 1)[1])
                snapshot(output, archive, names)
                hashes = {str((archive / name).relative_to(run_dir)): sha256(archive / name) for name in names}
                # 校准会覆盖根元数据，因此使用归档元数据，并另校验重算产物。
                hashes.update({str((output / name).relative_to(run_dir)): value for name, value in live_hashes.items()})
                record.update(status="completed", end_time=now(), artifact_hashes=hashes)
                write_json(run_dir / "run_meta.json", state)
            state.update(status="completed", end_time=now())
            write_json(run_dir / "run_meta.json", state)
            finish_report(run_dir)
        except BaseException as exc:
            for record in state["steps"].values():
                if record.get("status") == "running":
                    record.update(status="failed", end_time=now())
            state.update(status="failed", end_time=now(), error_type=type(exc).__name__)
            write_json(run_dir / "run_meta.json", state)
            print(f"[中断] 重复同一命令即可续跑；日志和已有结果保留在 {run_dir}", flush=True)
            raise
        print(f"[完成] {run_dir / 'summary.md'}\n[结果包] {run_dir / 'results.zip'}", flush=True)
        return run_dir


def main():
    args = parse_args(add_arguments)
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    run_comparison(args, cfg)


if __name__ == "__main__":
    main()
