"""C18/C20 三组对照入口的 CPU 离线编排和产物校验测试。"""
import argparse
import csv
import importlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    module = importlib.import_module("e0_refilter_compare")
    monkeypatch.setattr(module, "load_trait", lambda name, split: {
        "questions": [f"{name} 问题 {i}" for i in range(10)]})
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_refilter(script, output, paired):
    """重算只写假向量字节；本测试验证哈希、依赖和归档，不执行模型。"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "directions.pt").write_bytes(b"paired-vector" if paired else b"cohfilt-vector")
    counts = {"n_total": 20, "n_trait_pairs": 8, "n_kept_pairs": 8 if paired else 6}
    summary = {"paired_only": paired, "poles": {p: counts for p in script.DEFAULT_POLES},
               "cn_cosine_before": [-0.7] * 17, "cn_cosine_after": [-0.5] * 17}
    write_json(output / "refilter_summary.json", summary)
    (output / "refilter_samples.jsonl").write_text('{"kept":true}\n', encoding="utf-8")
    write_json(output / "vectors" / "refilter_meta.json", {"task": "C18", "summary": summary})
    hashes = {p.relative_to(output).as_posix(): script.sha256(p)
              for p in output.rglob("*") if p.is_file()}
    write_json(output / "run_meta.json", {
        "task": "C18", "status": "completed", "end_time": "2026-10-03T01:00:00Z",
        "paired_only": paired, "judge_called": False, "generation_performed": False,
        "artifact_hashes": hashes})


def write_evaluation(script, output, measurement):
    dim, pole, coefs = script.MEASUREMENTS[measurement]
    name = script.BIG_FIVE[dim][0 if pole == "high" else 1]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for coef in coefs:
        samples = [{"q": q, "a": "已理解题目并作答", "trait": 60.0, "coherence": 90.0,
                    "task": 80.0, "n_generated_tokens": 123, "finish_reason": "eos"}
                   for q in script.load_trait(name, "eval")["questions"] for _ in range(3)]
        records.append({"layer": 16, "dim": dim, "pole": pole, "coef": coef,
                        "trait_name": name, "n_samples": 3, "max_new_tokens": 320,
                        "with_task": True, "result": {
                            "trait": 60.0, "coherence": 90.0, "task": 80.0,
                            "n_valid": 30, "n_valid_coherence": 30, "n_valid_task": 30,
                            "samples": samples}})
    (output / script.EVAL_FILES[1]).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    with (output / script.EVAL_FILES[0]).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "dim", "coef", "trait", "coherence", "pole", "task"])
        writer.writeheader()
        for row in records:
            writer.writerow({"layer": 16, "dim": dim, "coef": row["coef"], "trait": 60,
                             "coherence": 90, "pole": pole, "task": 80})
    write_json(output / "run_meta.json", {
        "status": "completed", "end_time": "2026-10-03T02:00:00Z", "layers": [16],
        "dims": [dim], "pole": pole, "coefs": coefs, "n_samples": 3,
        "with_task": True, "max_new_tokens": 320,
        "task_score_validation": "unvalidated", "task_score_use": "exploratory",
        "data_hashes": {str(output / "directions.pt"): script.sha256(output / "directions.pt")}})
    return records


@pytest.fixture
def runtime(script, monkeypatch, tmp_path):
    source = tmp_path / "main"
    source.mkdir()
    (source / "directions.pt").write_bytes(b"original-vector")
    (source / "calibration_judge.csv").write_bytes(b"historical-results\r\n")
    (source / "judge_cache.jsonl").write_bytes(b"historical-cache\n")
    write_json(source / "run_meta.json", {"status": "original"})
    config = tmp_path / "config.yaml"
    cfg = {"out_path": source, "out_dir": str(source), "model": "offline/model",
           "max_new_tokens": 200, "calib_questions": 10}
    config.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "out_path"}), encoding="utf-8")
    args = argparse.Namespace(config=str(config), fresh=False)
    plan = {"fingerprint": "fixed-inputs", "provenance": {"expected_generated_answers": 810}}
    monkeypatch.setattr(script, "build_plan", lambda *args: dict(plan))
    calls, commands = [], []
    control = {"fail": None, "omit": None}

    def run(command, log_path):
        config_path = Path(command[command.index("--config") + 1])
        if "--source-out" in command:
            output = Path(command[command.index("--output-dir") + 1])
            step = f"refilter_{output.name}"
        else:
            output = Path(yaml.safe_load(config_path.read_text(encoding="utf-8"))["out_dir"])
            dim, pole = command[command.index("--dims") + 1], command[command.index("--pole") + 1]
            step = f"{output.name}_{dim.lower()}_{pole}"
        calls.append(step)
        commands.append(command)
        Path(log_path).write_text(f"离线执行 {step}\n", encoding="utf-8")
        if control["fail"] == step:
            raise RuntimeError("模拟实验中断")
        if "--source-out" in command:
            write_refilter(script, output, "--paired-only" in command)
        else:
            write_evaluation(script, output, step.split("_", 1)[1])
        if control["omit"] == step:
            (output / "run_meta.json").unlink()

    monkeypatch.setattr(script, "run_step", run)
    return args, cfg, plan, calls, commands, control


def expected_steps(script):
    return ["refilter_paired", "refilter_cohfilt"] + [
        f"{variant}_{measurement}" for variant in script.VARIANTS for measurement in script.MEASUREMENTS]


def test_full_grid_source_preservation_separate_outputs_and_report(script, runtime):
    args, cfg, _, calls, commands, _ = runtime
    source = cfg["out_path"]
    originals = {p: p.read_bytes() for p in source.iterdir() if p.is_file()}
    run_dir = script.run_comparison(args, cfg)
    assert calls == expected_steps(script) and len(calls) == 14
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    assert read_json(run_dir / "run_meta.json")["status"] == "completed"
    total = 0
    for variant in script.VARIANTS:
        config = yaml.safe_load((run_dir / "configs" / f"{variant}.yaml").read_text(encoding="utf-8"))
        assert config["max_new_tokens"] == 200
        assert Path(config["out_dir"]) == run_dir / "variants" / variant
        for measurement in script.MEASUREMENTS:
            path = run_dir / "steps" / f"{variant}_{measurement}" / script.EVAL_FILES[1]
            total += sum(len(json.loads(line)["result"]["samples"]) for line in path.read_text(encoding="utf-8").splitlines())
    assert total == 810
    for command in commands[:2]:
        assert "--max-new-tokens" not in command
        assert Path(command[command.index("--config") + 1]) == Path(args.config)
    for command in commands[2:]:
        assert command[command.index("--max-new-tokens") + 1] == "320"
        assert "--with-task" in command
    assert len({(run_dir / "variants" / variant / "directions.pt").read_bytes() for variant in script.VARIANTS}) == 3
    with (run_dir / "task_blind.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["id", "question", "answer", "task_human"]
        assert len(list(reader)) == 30
    assert "探索" in (run_dir / "summary.md").read_text(encoding="utf-8")
    with zipfile.ZipFile(run_dir / "results.zip") as bundle:
        assert bundle.testzip() is None
        assert "summary.md" in bundle.namelist() and "task_blind.csv" in bundle.namelist()
        assert not any("cache" in n or n.endswith(".pt") for n in bundle.namelist())


def test_second_identical_run_has_no_subprocesses(script, runtime):
    args, cfg, _, calls, _, _ = runtime
    run_dir = script.run_comparison(args, cfg)
    calls.clear()
    assert script.run_comparison(args, cfg) == run_dir
    assert calls == []


def test_corrupt_cohfilt_direction_only_rebuilds_its_branch(script, runtime):
    args, cfg, _, calls, _, _ = runtime
    run_dir = script.run_comparison(args, cfg)
    before = (run_dir / "steps" / "paired_c_high" / "run_meta.json").read_bytes()
    (run_dir / "variants" / "cohfilt" / "directions.pt").write_bytes(b"damaged")
    calls.clear()
    assert script.run_comparison(args, cfg) == run_dir
    assert calls == ["refilter_cohfilt"] + [f"cohfilt_{m}" for m in script.MEASUREMENTS]
    assert (run_dir / "steps" / "paired_c_high" / "run_meta.json").read_bytes() == before


def test_corrupt_evaluation_snapshot_only_remeasures_that_step(script, runtime):
    args, cfg, _, calls, _, _ = runtime
    run_dir = script.run_comparison(args, cfg)
    (run_dir / "steps" / "paired_n_high" / "calibration_judge.csv").write_text("损坏的结果", encoding="utf-8")
    calls.clear()
    assert script.run_comparison(args, cfg) == run_dir
    assert calls == ["paired_n_high"]


@pytest.mark.parametrize("failed_step", ["refilter_cohfilt", "paired_c_low"])
def test_failure_stops_and_resumes_completed_dependencies(script, runtime, failed_step):
    args, cfg, _, calls, _, control = runtime
    control["fail"] = failed_step
    with pytest.raises(RuntimeError, match="模拟实验中断"):
        script.run_comparison(args, cfg)
    steps = expected_steps(script)
    failed_index = steps.index(failed_step)
    assert calls == steps[:failed_index + 1]
    latest = read_json(cfg["out_path"] / "refilter_diagnostics" / "latest.json")
    run_dir = cfg["out_path"] / "refilter_diagnostics" / latest["run_id"]
    assert read_json(run_dir / "run_meta.json")["status"] == "failed"
    control["fail"] = None
    calls.clear()
    assert script.run_comparison(args, cfg) == run_dir
    assert calls == steps[failed_index:]
    assert read_json(run_dir / "run_meta.json")["status"] == "completed"


@pytest.mark.parametrize("step", ["refilter_paired", "original_c_high"])
def test_exit_zero_missing_required_output_stops(script, runtime, step):
    args, cfg, _, calls, _, control = runtime
    control["omit"] = step
    with pytest.raises((FileNotFoundError, RuntimeError)):
        script.run_comparison(args, cfg)
    assert calls[-1] == step
    assert len(calls) == expected_steps(script).index(step) + 1


@pytest.mark.parametrize("change", ["fingerprint", "fresh"])
def test_changed_dependency_or_fresh_creates_separate_run(script, runtime, change):
    args, cfg, plan, calls, _, _ = runtime
    old = script.run_comparison(args, cfg)
    old_meta = (old / "run_meta.json").read_bytes()
    if change == "fingerprint":
        plan["fingerprint"] = "changed-source"
    else:
        args.fresh = True
    calls.clear()
    new = script.run_comparison(args, cfg)
    assert new != old and calls == expected_steps(script)
    assert (old / "run_meta.json").read_bytes() == old_meta


def test_source_changed_during_subprocess_stops_before_archive(script, runtime, monkeypatch):
    args, cfg, plan, calls, _, _ = runtime
    source = cfg["out_path"] / "directions.pt"
    plan["provenance"]["file_hashes"] = {str(source): script.sha256(source)}
    run = script.run_step

    def mutate_after_run(command, log_path):
        run(command, log_path)
        source.write_bytes(b"external-change-during-experiment")

    monkeypatch.setattr(script, "run_step", mutate_after_run)
    with pytest.raises(RuntimeError, match="运行期间输入发生变化"):
        script.run_comparison(args, cfg)
    assert calls == ["refilter_paired"]
    latest = read_json(cfg["out_path"] / "refilter_diagnostics" / "latest.json")
    run_dir = cfg["out_path"] / "refilter_diagnostics" / latest["run_id"]
    state = read_json(run_dir / "run_meta.json")
    assert state["status"] == "failed" and state["steps"]["refilter_paired"]["status"] == "failed"
    assert not (run_dir / "steps" / "refilter_paired").exists()


@pytest.mark.parametrize("bad", ["denominator", "missing_score", "question", "nan", "tokens", "finish_reason", "csv"])
def test_evaluation_rejects_bad_fields_counts_or_questions(script, tmp_path, bad):
    output = tmp_path / "measurement"
    output.mkdir()
    (output / "directions.pt").write_bytes(b"vector")
    records = write_evaluation(script, output, "c_high")
    script.validate_evaluation(output, "c_high")
    result, sample = records[0]["result"], records[0]["result"]["samples"][0]
    if bad == "denominator":
        result["n_valid_task"] = 29
    elif bad == "missing_score":
        del sample["coherence"]
    elif bad == "question":
        sample["q"] = "来自其他极点或test的问题"
    elif bad == "nan":
        sample["task"] = float("nan")
    elif bad == "tokens":
        sample["n_generated_tokens"] = True
    elif bad == "finish_reason":
        sample["finish_reason"] = "length"
    else:
        path = output / "calibration_judge.csv"
        path.write_text(path.read_text(encoding="utf-8").replace(",60,90,", ",61,90,", 1), encoding="utf-8")
    (output / script.EVAL_FILES[1]).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises((RuntimeError, KeyError)):
        script.validate_evaluation(output, "c_high")


def test_refilter_rejects_outside_artifact_even_with_matching_hash(script, tmp_path):
    output = tmp_path / "cohfilt"
    write_refilter(script, output, False)
    script.validate_refilter(output, False)
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"not-an-artifact")
    meta = read_json(output / "run_meta.json")
    meta["artifact_hashes"]["../private.txt"] = script.sha256(outside)
    write_json(output / "run_meta.json", meta)
    with pytest.raises(RuntimeError, match="产物校验失败"):
        script.validate_refilter(output, False)


@pytest.mark.parametrize("outside", ["source", "destination", "root"])
def test_preserve_never_moves_outside_or_whole_run(script, tmp_path, outside):
    root = tmp_path / "run"
    root.mkdir()
    src = root / "result.txt"
    src.write_bytes(b"protected")
    destination = root / "previous" / "result.txt"
    if outside == "source":
        src = tmp_path / "source.txt"
        src.write_bytes(b"protected")
    elif outside == "destination":
        destination = tmp_path / "outside.txt"
    else:
        src = root
    with pytest.raises(ValueError, match="本轮目录内"):
        script.preserve(src, destination, root)
    assert src.exists() and not destination.exists()
