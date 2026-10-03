"""一键诊断编排的离线测试：假子进程不加载模型、不生成回答、不调用评委。"""
import argparse
import importlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("e0_diagnostics")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_outputs(script, out, step, marker):
    """构造两步各自完整的假产物，让归档、恢复和压缩走真实文件操作。"""
    out.mkdir(parents=True, exist_ok=True)
    meta = {"status": "completed", "marker": marker, "task": "C17" if step == "c17" else "C6"}
    (out / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if step == "c17":
        group = {"n_total": 1, "n_valid": 1, "n_missing": 0, "mean_coherence": 80,
                 "below_threshold_fraction": 0, "n_below_threshold": 0}
        summary = {"threshold": 50, "poles": {
            p: {"pos": group, "neg": group}
            for p in ("dependable", "careless", "nervous", "calm")}}
        (out / "extract_coherence.json").write_text(json.dumps(summary), encoding="utf-8")
        (out / "extract_coherence_samples.jsonl").write_text(
            json.dumps({"marker": marker, "coherence": 80}) + "\n", encoding="utf-8")
    else:
        csv = "layer,dim,coef,trait,coherence,pole\n"
        records = []
        for dim in ("C", "N"):
            for coef in (-0.2, 0.0):
                csv += f"16,{dim},{coef},50,90,low\n"
                records.append({"layer": 16, "dim": dim, "coef": coef, "pole": "low",
                                "n_samples": 3, "marker": marker,
                                "result": {"trait": 50, "coherence": 90, "samples": []}})
        (out / "calibration_judge.csv").write_text(csv, encoding="utf-8")
        (out / "calibration_judge_samples.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def runtime(script, monkeypatch, tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("model: offline-model\n", encoding="utf-8")
    args = argparse.Namespace(config=str(config), fresh=False)
    cfg = {"out_path": out, "calib_questions": 10}
    plan = {"fingerprint": "同一批输入", "provenance": {"config": str(config)}}
    monkeypatch.setattr(script, "build_plan", lambda *a: dict(plan))
    monkeypatch.setattr(script, "validate_step", lambda *a: None)
    commands = script.build_commands(args.config)
    calls = []
    control = {"fail": None}

    def run(command, log_path):
        step = next(name for name, expected in commands.items() if command == expected)
        calls.append(step)
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"离线运行 {step}\n", encoding="utf-8")
        if step == control["fail"]:
            (out / "run_meta.json").write_text('{"status":"failed"}', encoding="utf-8")
            raise RuntimeError("模拟子进程失败")
        write_outputs(script, out, step, len(calls))

    monkeypatch.setattr(script, "run_step", run)
    return args, cfg, plan, calls, control


def test_default_plan_keeps_fixed_experiment_settings(script):
    commands = script.build_commands("configs/main.yaml")
    assert list(commands) == ["c17", "c6_low"]
    for command in commands.values():
        assert command[0] == sys.executable
        assert command[command.index("--config") + 1] == "configs/main.yaml"
    c17 = commands["c17"]
    assert "-u" in c17 and any(Path(v).name == "e0_extract_coherence.py" for v in c17)
    if "--poles" in c17:
        assert c17[c17.index("--poles") + 1:] == ["dependable", "careless", "nervous", "calm"]
    c6 = commands["c6_low"]
    assert "-u" in c6 and any(Path(v).name == "e0_calibrate.py" for v in c6)
    assert c6[c6.index("--layers") + 1] == "16"
    assert c6[c6.index("--dims") + 1:c6.index("--dims") + 3] == ["C", "N"]
    assert [float(v) for v in c6[c6.index("--coefs") + 1:c6.index("--coefs") + 3]] == [-0.2, 0]
    assert c6[c6.index("--pole") + 1] == "low"
    assert c6[c6.index("--n-samples") + 1] == "3"
    assert "--save-samples" in c6 and "--reuse-baseline" in c6


def test_full_run_preserves_previous_outputs_and_separate_metadata(script, runtime):
    args, cfg, _, calls, _ = runtime
    out = cfg["out_path"]
    names = {name for outputs in script.STEPS.values() for name in outputs}
    previous = {name: f"旧结果 {name}\r\n".encode("utf-8") for name in names}
    for name, data in previous.items():
        (out / name).write_bytes(data)
    vector_dir = out / "vectors"
    vector_dir.mkdir()
    source = vector_dir / "dependable_samples.json"
    source.write_bytes(b"original-source\r\n")

    run_dir = script.run_diagnostics(args, cfg)

    assert calls == ["c17", "c6_low"]
    assert source.read_bytes() == b"original-source\r\n"
    for name, data in previous.items():
        assert (run_dir / "backup" / name).read_bytes() == data
    meta = read_json(run_dir / "run_meta.json")
    assert meta["status"] == "completed"
    for step in script.STEPS:
        assert meta["steps"][step]["status"] == "completed"
        assert set(meta["steps"][step]["artifact_hashes"]) == set(script.STEPS[step])
        assert all((run_dir / step / name).is_file() for name in script.STEPS[step])
    assert read_json(run_dir / "c17" / "run_meta.json")["marker"] == 1
    assert read_json(run_dir / "c6_low" / "run_meta.json")["marker"] == 2
    assert (run_dir / "summary.md").is_file()
    with zipfile.ZipFile(run_dir / "results.zip") as bundle:
        assert bundle.testzip() is None
        assert any(name.endswith("c17/extract_coherence.json") for name in bundle.namelist())
        assert any(name.endswith("c6_low/calibration_judge.csv") for name in bundle.namelist())


def test_completed_identical_run_skips_paid_work(script, runtime):
    args, cfg, _, calls, _ = runtime
    run_dir = script.run_diagnostics(args, cfg)
    archived = (run_dir / "c17" / "run_meta.json").read_bytes()
    assert script.run_diagnostics(args, cfg) == run_dir
    assert calls == ["c17", "c6_low"]
    assert (run_dir / "c17" / "run_meta.json").read_bytes() == archived


@pytest.mark.parametrize("failed_step", ["c17", "c6_low"])
def test_failure_stops_and_retry_only_runs_unfinished_steps(script, runtime, failed_step):
    args, cfg, _, calls, control = runtime
    control["fail"] = failed_step
    with pytest.raises(RuntimeError, match="模拟子进程失败"):
        script.run_diagnostics(args, cfg)
    initial_calls = ["c17"] if failed_step == "c17" else ["c17", "c6_low"]
    assert calls == initial_calls
    latest = read_json(cfg["out_path"] / "diagnostics" / "latest.json")
    run_dir = cfg["out_path"] / "diagnostics" / latest["run_id"]
    failed = read_json(run_dir / "run_meta.json")
    assert failed["status"] == "failed"
    assert failed["steps"][failed_step]["status"] == "failed"
    if failed_step == "c6_low":
        assert failed["steps"]["c17"]["status"] == "completed"
    control["fail"] = None
    assert script.run_diagnostics(args, cfg) == run_dir
    assert calls == initial_calls + (["c17", "c6_low"] if failed_step == "c17" else ["c6_low"])
    assert read_json(run_dir / "run_meta.json")["status"] == "completed"


@pytest.mark.parametrize("force_new", ["fingerprint", "fresh"])
def test_changed_inputs_or_fresh_start_new_run(script, runtime, force_new):
    args, cfg, plan, calls, _ = runtime
    old = script.run_diagnostics(args, cfg)
    old_meta = (old / "run_meta.json").read_bytes()
    if force_new == "fingerprint":
        plan["fingerprint"] = "输入已改变"
    else:
        args.fresh = True
    new = script.run_diagnostics(args, cfg)
    assert new != old
    assert calls == ["c17", "c6_low", "c17", "c6_low"]
    assert (old / "run_meta.json").read_bytes() == old_meta


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_damaged_completed_archive_is_never_silently_reused(script, runtime, damage):
    args, cfg, _, calls, _ = runtime
    run_dir = script.run_diagnostics(args, cfg)
    path = run_dir / "c17" / "extract_coherence.json"
    if damage == "missing":
        path.unlink()
    else:
        path.write_text('{"unexpected":"内容改变"}', encoding="utf-8")
    try:
        script.run_diagnostics(args, cfg)
    except (ValueError, RuntimeError, FileNotFoundError):
        pass
    else:
        assert len(calls) > 2, "损坏的完成产物不得直接跳过付费步骤并宣告成功"


def test_preflight_failure_does_not_execute_or_overwrite(script, runtime, monkeypatch):
    args, cfg, _, calls, _ = runtime
    previous = cfg["out_path"] / "calibration_judge.csv"
    previous.write_bytes(b"existing-results\r\n")

    def fail(*args):
        raise FileNotFoundError("缺少正式提取回答")

    monkeypatch.setattr(script, "build_plan", fail)
    with pytest.raises(FileNotFoundError, match="缺少正式提取回答"):
        script.run_diagnostics(args, cfg)
    assert calls == []
    assert previous.read_bytes() == b"existing-results\r\n"
    assert not list((cfg["out_path"] / "diagnostics").glob("*/run_meta.json"))


def test_exit_zero_with_invalid_outputs_is_failure(script, runtime, monkeypatch):
    args, cfg, _, calls, _ = runtime

    def reject(step, out_path, cfg):
        raise ValueError("产物校验失败")

    monkeypatch.setattr(script, "validate_step", reject)
    with pytest.raises(ValueError, match="产物校验失败"):
        script.run_diagnostics(args, cfg)
    assert calls == ["c17"]
    latest = read_json(cfg["out_path"] / "diagnostics" / "latest.json")
    meta = read_json(cfg["out_path"] / "diagnostics" / latest["run_id"] / "run_meta.json")
    assert meta["status"] == "failed" and meta["steps"]["c17"]["status"] == "failed"


def test_output_lock_rejects_concurrency_and_releases_after_error(script, tmp_path):
    with pytest.raises(ValueError, match="主动结束"):
        with script.output_lock(tmp_path):
            with pytest.raises((RuntimeError, BlockingIOError, OSError)):
                with script.output_lock(tmp_path):
                    pytest.fail("同一输出目录不能被两个编排器同时持有")
            raise ValueError("主动结束")
    with script.output_lock(tmp_path):
        pass


@pytest.fixture
def real_inputs(script, monkeypatch, tmp_path):
    """创建真实预检输入；仅保存 CPU 张量，评委构造函数不会请求网络。"""
    out = tmp_path / "output"
    (out / "vectors").mkdir(parents=True)
    config = tmp_path / "test.yaml"
    config.write_text("model: unavailable/offline-model\ncalib_questions: 10\n", encoding="utf-8")
    args = argparse.Namespace(config=str(config), fresh=False)
    cfg = {"out_path": out, "model": "unavailable/offline-model", "calib_questions": 10,
           "use_judge": True}
    for pole in script.DEFAULT_POLES:
        rows = [{"side": side, "question": "问题", "answer": "已有回答", "score": score}
                for side, score in (("pos", 90), ("neg", 10))]
        (out / "vectors" / f"{pole}_samples.json").write_text(json.dumps(rows), encoding="utf-8")
    directions = {"U": torch.eye(5, 8).repeat(17, 1, 1), "s": torch.ones(17),
                  "dims": ["O", "C", "E", "A", "N"]}
    torch.save(directions, out / "directions.pt")
    monkeypatch.setenv("JUDGE_MODEL", "固定离线评委")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "禁止出现在元数据的测试令牌")
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "16")
    monkeypatch.setenv("JUDGE_EXTRA_BODY", '{"thinking":{"type":"disabled"}}')
    return args, cfg, directions


def test_real_plan_fingerprints_inputs_and_judge_but_not_secret(script, real_inputs, monkeypatch):
    args, cfg, _ = real_inputs
    plan = script.build_plan(args, cfg)
    assert len(plan["fingerprint"]) == 64
    assert "禁止出现在元数据的测试令牌" not in json.dumps(plan, ensure_ascii=False, default=str)
    hashes = plan["provenance"]["file_hashes"]
    assert str((cfg["out_path"] / "directions.pt").resolve()) in hashes
    assert str(Path(args.config).resolve()) in hashes
    assert script.build_plan(args, cfg)["fingerprint"] == plan["fingerprint"]
    monkeypatch.setenv("JUDGE_API_KEY", "轮换后的令牌")
    assert script.build_plan(args, cfg)["fingerprint"] == plan["fingerprint"]
    monkeypatch.setenv("JUDGE_MODEL", "另一评委")
    assert script.build_plan(args, cfg)["fingerprint"] != plan["fingerprint"]
    monkeypatch.setenv("JUDGE_MODEL", "固定离线评委")
    source = cfg["out_path"] / "vectors" / "careless_samples.json"
    rows = read_json(source)
    rows[0]["answer"] = "修改后的回答"
    source.write_text(json.dumps(rows), encoding="utf-8")
    assert script.build_plan(args, cfg)["fingerprint"] != plan["fingerprint"]


@pytest.mark.parametrize("invalid", ["missing", "shape", "nonfinite", "dims", "environment"])
def test_real_preflight_rejects_invalid_inputs_before_work(script, real_inputs, monkeypatch, invalid):
    args, cfg, directions = real_inputs
    path = cfg["out_path"] / "directions.pt"
    if invalid == "missing":
        path.unlink()
    elif invalid == "shape":
        directions["U"] = directions["U"][:16]
        torch.save(directions, path)
    elif invalid == "nonfinite":
        directions["U"][16, 0, 0] = float("nan")
        torch.save(directions, path)
    elif invalid == "dims":
        directions["dims"] = ["C", "O", "E", "A", "N"]
        torch.save(directions, path)
    else:
        monkeypatch.delenv("JUDGE_API_KEY")
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        script.build_plan(args, cfg)
    assert not (cfg["out_path"] / "run_meta.json").exists()


def valid_result(script, cfg, step):
    out = cfg["out_path"]
    write_outputs(script, out, step, 1)
    meta = read_json(out / "run_meta.json")
    meta["end_time"] = "2026-10-03T00:00:00+00:00"
    if step == "c17":
        sources = script.load_inputs(out, script.DEFAULT_POLES)
        records = [{"pole": pole, "source_index": index, "source_sha256": source["source_sha256"],
                    "coherence": 80}
                   for pole, source in sources.items() for index in range(len(source["samples"]))]
    else:
        meta.update(layers=[16], dims=["C", "N"], coefs=[-0.2, 0], pole="low", n_samples=3)
        records = [json.loads(line) for line in (out / script.STEPS[step][1]).read_text(encoding="utf-8").splitlines()]
        for record in records:
            record["result"]["samples"] = [{"q": f"问题{i}", "a": "已有回答", "trait": 50} for i in range(30)]
    (out / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / script.STEPS[step][1]).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return meta, records


@pytest.mark.parametrize("step", ["c17", "c6_low"])
def test_real_validation_accepts_complete_outputs_and_rejects_partial_metadata(script, real_inputs, step):
    _, cfg, _ = real_inputs
    meta, _ = valid_result(script, cfg, step)
    script.validate_step(step, cfg["out_path"], cfg)
    meta["status"] = "running"
    (cfg["out_path"] / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="未记录完成"):
        script.validate_step(step, cfg["out_path"], cfg)


@pytest.mark.parametrize("step", ["c17", "c6_low"])
def test_real_validation_rejects_misaligned_or_incomplete_samples(script, real_inputs, step):
    _, cfg, _ = real_inputs
    _, records = valid_result(script, cfg, step)
    if step == "c17":
        records[0]["source_sha256"] = "不是本轮来源"
    else:
        records[0]["result"]["samples"].pop()
    path = cfg["out_path"] / script.STEPS[step][1]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(RuntimeError):
        script.validate_step(step, cfg["out_path"], cfg)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_subprocess_preserves_log_and_failure_code(script, tmp_path, exit_code):
    log = tmp_path / "subprocess.log"
    command = [sys.executable, "-c", f"import sys; print('离线日志'); sys.exit({exit_code})"]
    if exit_code:
        with pytest.raises(RuntimeError, match="退出码 7"):
            script.run_step(command, log)
    else:
        script.run_step(command, log)
    assert "离线日志" in log.read_text(encoding="utf-8")


@pytest.mark.parametrize("field,value", [("dtype", "invalid"), ("max_new_tokens", 0),
                                         ("max_new_tokens", True), ("max_new_tokens", 1.5)])
def test_real_plan_rejects_invalid_generation_config(script, real_inputs, field, value):
    args, cfg, _ = real_inputs
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        script.build_plan(args, cfg)


def test_local_missing_model_shard_fails_before_tokenizer_or_judge(script, real_inputs, monkeypatch, tmp_path):
    args, cfg, _ = real_inputs
    monkeypatch.setattr(script, "ROOT", tmp_path)
    cfg["model"] = "models/test-model"
    model = tmp_path / cfg["model"]
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 17, "hidden_size": 8}), encoding="utf-8")
    (model / "model-00001-of-00002.safetensors").write_bytes(b"fake-weight-shard")
    index = {"weight_map": {"first": "model-00001-of-00002.safetensors",
                            "second": "model-00002-of-00002.safetensors"}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("模型分片缺失时应在构造分词器或评委前失败")

    monkeypatch.setattr(script.transformers.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(script, "Judge", forbidden)
    with pytest.raises(FileNotFoundError, match="分片缺失"):
        script.build_plan(args, cfg)
