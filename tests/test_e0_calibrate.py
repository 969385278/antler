"""C2/C6：用 CPU 微型模型检查扫描、测量极点与基线复用，不下载模型或调用 API。"""
import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

from test_core import model  # 复用已有随机微型模型 fixture
from antler.core.hooks import Steer, no_steer


@pytest.fixture
def calibration(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("e0_calibrate")


def arguments(calibration, *argv):
    parser = argparse.ArgumentParser()
    calibration.add_arguments(parser)
    return parser.parse_args(argv)


def test_legacy_defaults_and_cli_overrides(calibration):
    cfg = {"layers": [0, 2], "coefs": [-0.1, 0, 0.1]}
    args = arguments(calibration)
    assert calibration.scan_settings(args, cfg, 4) == ([0, 2], [-0.1, 0, 0.1])
    assert not args.save_samples and not args.reuse_baseline and args.n_samples == 1
    assert args.pole == "high"
    assert args.with_task is False and args.max_new_tokens is None
    args = arguments(calibration, "--layers", "1", "3", "--coefs", "-0.4", "0", "0.4",
                     "--n-samples", "3")
    assert calibration.scan_settings(args, cfg, 4) == ([1, 3], [-0.4, 0, 0.4])
    assert cfg == {"layers": [0, 2], "coefs": [-0.1, 0, 0.1]}


@pytest.mark.parametrize("argv", [
    ["--layers", "-1"], ["--layers", "4"], ["--layers", "1", "1"],
    ["--coefs", "nan"], ["--coefs", "0", "0"], ["--n-samples", "0"],
    ["--dims", "X"], ["--mode", "mcq", "--save-samples"],
    ["--mode", "mcq", "--n-samples", "2"],
    ["--mode", "mcq", "--pole", "low"], ["--mode", "mcq", "--pole", "both"],
    ["--mode", "mcq", "--with-task"], ["--mode", "mcq", "--max-new-tokens", "320"],
    ["--max-new-tokens", "0"], ["--max-new-tokens", "-1"],
])
def test_invalid_settings(calibration, argv):
    with pytest.raises(ValueError):
        calibration.scan_settings(arguments(calibration, *argv), {"layers": [1]}, 4)


@pytest.mark.parametrize("reuse,save", [(False, False), (True, True)])
@pytest.mark.parametrize("pole", ["high", "low", "both"])
def test_judge_sweep_outputs(calibration, model, monkeypatch, tmp_path, reuse, save, pole):
    cfg = {"out_path": tmp_path, "layers": [1, 2], "coefs": [-0.1, 0, 0.1],
           "calib_questions": 2, "max_new_tokens": 8}
    monkeypatch.setattr(calibration, "setup", lambda *a, **kw: (cfg, model, None, object()))
    U = torch.eye(5, model.config.hidden_size).repeat(model.config.num_hidden_layers, 1, 1)
    monkeypatch.setattr(calibration, "load_directions", lambda p: {"U": U, "s": torch.ones(4)})
    calls = []

    def measure(m, tok, judge, dim, **kw):
        calls.append((dim, kw))
        # 实际执行微型模型和注入，确认本任务仍走 Steer 且运行后卸载。
        with torch.no_grad(), (Steer(m, kw["steer"]) if kw["steer"] else no_steer()):
            value = float(m(input_ids=torch.tensor([[1, 2, 3]])).logits.mean())
        score = 60 if kw["pole"] == "high" else 20
        samples = [{"q": f"{dim}-{kw['pole']}-问题{q}", "a": f"回答{value}", "trait": score if q else None}
                   for _ in range(kw["n_samples"]) for q in range(kw["n_questions"])]
        return {"dim": dim, "pole": kw["pole"], "trait": score, "coherence": 90,
                "n_valid": kw["n_samples"], "samples": samples}

    monkeypatch.setattr(calibration, "trait_score", measure)
    argv = ["e0_calibrate.py", "--mode", "judge", "--dims", "O", "N", "--n-samples", "2", "--pole", pole]
    if reuse:
        argv += ["--reuse-baseline"]
    if save:
        argv += ["--save-samples"]
    monkeypatch.setattr(sys, "argv", argv)
    calibration.main()
    selected = ["high", "low"] if pole == "both" else [pole]
    assert len(calls) == (10 if reuse else 12) * len(selected)
    assert sum(kw["steer"] is None for _, kw in calls) == (2 if reuse else 4) * len(selected)
    assert all(kw["n_samples"] == 2 for _, kw in calls)
    assert all(kw["max_new_tokens"] == 8 and kw["with_task"] is False for _, kw in calls)
    assert {kw["pole"] for _, kw in calls} == set(selected)
    rows = list(csv.DictReader((tmp_path / "calibration_judge.csv").open(encoding="utf-8")))
    assert len(rows) == 12 * len(selected)
    assert list(rows[0]) == ["layer", "dim", "coef", "trait", "coherence", "pole"]
    assert all(float(r["trait"]) == (60 if r["pole"] == "high" else 20) for r in rows)
    path = tmp_path / "calibration_judge_samples.jsonl"
    assert path.exists() == save
    meta_path = tmp_path / "run_meta.json"
    assert meta_path.exists() == save
    if save:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["status"] == "completed" and meta["end_time"] is not None
        assert meta["layers"] == [1, 2] and meta["n_samples"] == 2
        assert len(meta["data_hashes"]) == 2 * len(selected)
        assert meta["pole"] == pole and meta["poles"] == selected
        assert meta["max_new_tokens"] == 8 and meta["with_task"] is False
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 12 * len(selected)
        reused = [r for r in records if r["baseline_reused"]]
        assert len(reused) == 2 * len(selected)
        assert all(r["coef"] == 0 and r["layer"] == 2 and r["source_layer"] == 1 for r in reused)
        assert all(len(r["result"]["samples"]) == 4 for r in records)
        assert all(r["pole"] == r["result"]["pole"] for r in records)
        assert all(r["result"]["samples"][0]["q"].startswith(f"{r['dim']}-{r['pole']}-") for r in reused)
        assert all(r["trait_name"] == calibration.BIG_FIVE[r["dim"]][0 if r["pole"] == "high" else 1]
                   for r in records)
        assert records[0]["result"]["samples"][0]["trait"] is None
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_missing_judge_does_not_truncate_output(calibration, model, monkeypatch, tmp_path):
    output = tmp_path / "calibration_judge.csv"
    output.write_text("已有结果\n", encoding="utf-8")
    monkeypatch.setattr(calibration, "setup", lambda *a, **kw: ({"out_path": tmp_path}, model, None, None))
    monkeypatch.setattr(sys, "argv", ["e0_calibrate.py"])
    with pytest.raises(ValueError, match="需要启用评委"):
        calibration.main()
    assert output.read_text(encoding="utf-8") == "已有结果\n"


def test_failed_run_records_status_without_credentials(calibration, monkeypatch, tmp_path):
    args = arguments(calibration, "--save-samples")
    args.config = "configs/main.yaml"
    cfg = {"out_path": tmp_path, "api_key": "配置中的凭据"}
    judge = argparse.Namespace(model="测试评委", base_url="https://example.invalid", max_tokens=16,
                               extra_body={"authorization": "嵌套凭据"}, api_key="不应写入的凭据")
    with pytest.raises(RuntimeError, match="模拟中断"):
        with calibration.record_run(args, cfg, judge, [1], [0]):
            raise RuntimeError("模拟中断")
    raw = (tmp_path / "run_meta.json").read_text(encoding="utf-8")
    meta = json.loads(raw)
    assert meta["status"] == "failed" and meta["end_time"] is not None
    assert "配置中的凭据" not in raw and "嵌套凭据" not in raw and "不应写入的凭据" not in raw


def test_mcq_baseline_reuse_preserves_csv(calibration, model, monkeypatch, tmp_path):
    cfg = {"out_path": tmp_path, "layers": [1, 2], "coefs": [-0.1, 0, 0.1]}
    monkeypatch.setattr(calibration, "setup", lambda *a, **kw: (cfg, model, None, None))
    U = torch.eye(5, model.config.hidden_size).repeat(4, 1, 1)
    monkeypatch.setattr(calibration, "load_directions", lambda p: {"U": U, "s": torch.ones(4)})
    monkeypatch.setattr(calibration, "load_jsonl", lambda p: [{"trait": "Openness"}, {"trait": "Neuroticism"}])
    calls = []

    def probabilities(m, tok, items, *, steer):
        calls.append((items, steer))
        return [0.75]

    monkeypatch.setattr(calibration, "mcq_high_prob", probabilities)
    monkeypatch.setattr(sys, "argv", ["e0_calibrate.py", "--mode", "mcq", "--dims", "O", "N", "--reuse-baseline"])
    calibration.main()
    assert len(calls) == 10
    rows = list(csv.DictReader((tmp_path / "calibration_mcq.csv").open(encoding="utf-8")))
    assert len(rows) == 12
    assert all(r["trait"] == "75.0" and r["coherence"] == "" and r["pole"] == "high" for r in rows)
    assert not (tmp_path / "run_meta.json").exists()


def test_low_recommendation_uses_own_direction_and_coherence(calibration, model, monkeypatch, tmp_path, capsys):
    cfg = {"out_path": tmp_path, "layers": [1, 2], "coefs": [-0.2, 0.2]}
    monkeypatch.setattr(calibration, "setup", lambda *a, **kw: (cfg, model, None, object()))
    U = torch.eye(5, model.config.hidden_size).repeat(4, 1, 1)
    monkeypatch.setattr(calibration, "load_directions", lambda p: {"U": U, "s": torch.ones(4)})

    def measure(m, tok, judge, dim, *, steer, pole, **kw):
        layer, vector = next(iter(steer.items()))
        negative = vector[1] < 0
        # 第 1 层高极不连贯，但低极清楚地随负向增强；第 2 层低极增强较弱。
        if pole == "low":
            score = (90 if layer == 1 else 60) if negative else 20
            coherence = 90
        else:
            score = 20 if negative else 80
            coherence = 60 if layer == 1 else 90
        return {"trait": score, "coherence": coherence}

    monkeypatch.setattr(calibration, "trait_score", measure)
    monkeypatch.setattr(sys, "argv", ["e0_calibrate.py", "--dims", "C", "--pole", "both"])
    calibration.main()
    output = capsys.readouterr().out
    assert "C high: layer 2，效果 60.0" in output
    assert "C low: layer 1，效果 70.0" in output


@pytest.mark.parametrize("save_samples", [False, True])
def test_task_scoring_cli_csv_and_metadata(calibration, model, monkeypatch, tmp_path, save_samples):
    cfg = {"out_path": tmp_path, "layers": [1, 2], "coefs": [0], "max_new_tokens": 200}
    monkeypatch.setattr(calibration, "setup", lambda *a, **kw: (cfg, model, None, object()))
    monkeypatch.setattr(calibration, "load_directions", lambda p: {
        "U": torch.eye(5, model.config.hidden_size).repeat(4, 1, 1), "s": torch.ones(4)})
    calls = []

    def measure(*args, **kwargs):
        calls.append(kwargs)
        return {"trait": 25, "coherence": 85, "task": 90, "n_valid": 1,
                "n_valid_coherence": 1, "n_valid_task": 1,
                "samples": [{"q": "问题", "a": "回答", "trait": 25, "coherence": 85, "task": 90,
                             "n_generated_tokens": 320, "finish_reason": "length"}]}

    monkeypatch.setattr(calibration, "trait_score", measure)
    argv = ["e0_calibrate.py", "--dims", "C", "--pole", "low", "--reuse-baseline", "--with-task",
            "--max-new-tokens", "320"]
    if save_samples:
        argv.append("--save-samples")
    monkeypatch.setattr(sys, "argv", argv)
    calibration.main()
    assert len(calls) == 1 and calls[0]["with_task"] is True and calls[0]["max_new_tokens"] == 320
    assert cfg["max_new_tokens"] == 200
    rows = list(csv.DictReader((tmp_path / "calibration_judge.csv").open(encoding="utf-8")))
    assert list(rows[0]) == ["layer", "dim", "coef", "trait", "coherence", "pole", "task"]
    assert [r["task"] for r in rows] == ["90", "90"]
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["max_new_tokens"] == 320 and meta["with_task"] is True and meta["status"] == "completed"
    assert meta["task_score_validation"] == "unvalidated" and meta["task_score_use"] == "exploratory"
    assert set(meta["prompt_hashes"]) == {"COHERENCE_PROMPT", "TASK_PROMPT"}
    if save_samples:
        samples = [json.loads(line) for line in (tmp_path / "calibration_judge_samples.jsonl")
                   .read_text(encoding="utf-8").splitlines()]
        assert samples[1]["baseline_reused"] and samples[0]["result"] == samples[1]["result"]
        assert all(s["max_new_tokens"] == 320 and s["with_task"] is True for s in samples)


@pytest.mark.parametrize("limit", [0, -2, 1.5, True])
def test_invalid_config_generation_limit(calibration, limit):
    with pytest.raises(ValueError, match="正整数"):
        calibration.scan_settings(arguments(calibration), {"layers": [1], "max_new_tokens": limit}, 4)
