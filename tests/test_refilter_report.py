"""C18/C20：离线验证汇总、分层盲标和结果包文件边界。"""
import csv
import importlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("_refilter_report")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path, report):
    write_json(tmp_path / "run_meta.json", {"status": "completed", "task_judge_validated": False})
    for variant in report.VARIANTS:
        for measurement, (dim, pole, coefs) in report.MEASUREMENTS.items():
            directory = tmp_path / "steps" / f"{variant}_{measurement}"
            directory.mkdir(parents=True)
            records = []
            for coef in coefs:
                samples = [{"q": f"问题{i}", "a": f"回答 {variant} {measurement} {coef} {i}",
                            "trait": None if i == 0 else 40 + coef * 100,
                            "coherence": None if i < 2 else 90 + coef * 10,
                            "task": None if i < 3 else 80 + coef * 20,
                            "n_generated_tokens": 320 if i % 3 == 0 else 100,
                            "finish_reason": "length" if i % 3 == 0 else "eos"}
                           for i in range(30)]
                # 写入不可信的预汇总值，确保报告从逐条有效分重新统计。
                records.append({"layer": 16, "dim": dim, "pole": pole, "coef": coef,
                                "result": {"trait": 999, "coherence": 999, "task": 999, "samples": samples}})
            (directory / "calibration_judge_samples.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
            (directory / "calibration_judge.csv").write_text("layer,dim\n16,C\n", encoding="utf-8")
            write_json(directory / "run_meta.json", {"status": "completed"})
    for variant, kept, cosine in (("paired", 100, 0.6), ("cohfilt", 80, 0.3)):
        directory = tmp_path / "steps" / f"refilter_{variant}"
        write_json(directory / "refilter_summary.json", {
            "cn_cosine_before": [0.7] * 17, "cn_cosine_after": [cosine] * 17,
            "poles": {pole: {"n_total": 400, "n_pairs": 200, "n_trait_pairs": 100, "n_kept_pairs": kept}
                      for pole in ("dependable", "careless", "nervous", "calm")}})
        (directory / "refilter_samples.jsonl").write_text('{"kept": true}\n', encoding="utf-8")
    log = tmp_path / "attempts" / "first" / "log.txt"
    log.parent.mkdir(parents=True)
    log.write_text("日志\n", encoding="utf-8")
    return tmp_path


def test_summary_recomputes_separate_valid_means_and_baseline_deltas(report, run_dir):
    result = report.finish_report(run_dir)
    assert result == run_dir / "results.zip"
    text = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "| 原始向量 | C/low | -0.2 | 20.00 | -20.00 | 88.00 | -2.00 | 76.00 | -4.00 | 29/28/27 | 10/30 (33.3%) |" in text
    assert "999" not in text
    assert "| 仅配对过滤 | 0.70 | 0.60 |" in text
    assert "| 配对＋连贯性过滤 | 0.70 | 0.30 |" in text
    assert "| 配对＋连贯性过滤 | careless | 400 | 100 | 80 | 160 |" in text
    assert "不是三个独立种子实验" in text and "同系数不等于相同人格强度" in text
    assert "探索性诊断" in text and "Spearman ρ ≥ 0.6" in text


def test_missing_baseline_or_all_scores_remain_missing(report, run_dir):
    source = run_dir / "steps" / "original_c_high" / "calibration_judge_samples.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    for sample in records[0]["result"]["samples"]:
        sample["task"] = None
    source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    report.finish_report(run_dir)
    text = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "| 原始向量 | C/high | +0 | 40.00 | +0.00 | 90.00 | +0.00 | 缺失 | 缺失 | 29/28/0 |" in text
    assert "| 原始向量 | C/high | +0.2 | 60.00 | +20.00 | 92.00 | +2.00 | 84.00 | 缺失 |" in text


def test_blind_table_has_thirty_stratified_deterministic_rows(report, run_dir):
    report.finish_report(run_dir)
    csv_path = run_dir / "task_blind.csv"
    key_path = run_dir / "task_blind_key.json"
    before = (csv_path.read_bytes(), key_path.read_bytes())
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and len({r["id"] for r in rows}) == 30
    assert list(rows[0]) == ["id", "question", "answer", "task_human"]
    assert all(row["task_human"] == "" for row in rows)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert key["task_judge_validated"] is False
    assert key["n_unique_qa"] == 30 and key["n_duplicate_qa"] == 0
    mapping = key["samples"]
    assert set(mapping) == {r["id"] for r in rows}
    assert Counter(item["condition"]["variant"] for item in mapping.values()) == {
        "original": 10, "paired": 10, "cohfilt": 10}
    counts = Counter((item["condition"]["variant"], item["condition"]["measurement"],
                      item["condition"]["coef"]) for item in mapping.values())
    assert len(counts) == 27 and sum(n == 2 for n in counts.values()) == 3
    assert len({(item["source"], item["condition"]["coef"], item["sample_index"])
                for item in mapping.values()}) == 30
    assert [item["condition"]["variant"] for item in mapping.values()] != sorted(
        item["condition"]["variant"] for item in mapping.values())
    for row in rows:
        info = mapping[row["id"]]
        records = [json.loads(line) for line in (run_dir / info["source"]).read_text(encoding="utf-8").splitlines()]
        record = next(r for r in records if r["coef"] == info["condition"]["coef"])
        sample = record["result"]["samples"][info["sample_index"]]
        assert row["question"] == sample["q"] and row["answer"] == sample["a"]
        assert info["task_judge"] == sample["task"]
    report.finish_report(run_dir)
    assert before == (csv_path.read_bytes(), key_path.read_bytes())
    assert b"\r\n" not in csv_path.read_bytes()


def test_archive_excludes_models_cache_config_secrets_and_previous(report, run_dir):
    forbidden = [".env", "directions.pt", "judge_cache.jsonl", "secret.json", "config.json",
                 "steps/original_c_high/secret.json", "steps/original_c_high/config.json",
                 "steps/original_c_high/judge_cache.jsonl", "steps/original_c_high/directions.pt",
                 "steps/original_c_high/previous/calibration_judge.csv",
                 "steps/config/run_meta.json", "steps/secret/run_meta.json",
                 "previous/run_meta.json", "attempts/previous/log.txt", "attempts/first/secret.json"]
    for relative in forbidden:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("不应分享的文件", encoding="utf-8")
    archive = report.finish_report(run_dir)
    with zipfile.ZipFile(archive) as packed:
        names = set(packed.namelist())
        assert not names.intersection(forbidden)
        assert {"summary.md", "run_meta.json", "task_blind.csv", "task_blind_key.json",
                "task_blind_instructions.md",
                "attempts/first/log.txt", "steps/refilter_paired/refilter_summary.json",
                "steps/original_c_high/calibration_judge_samples.jsonl"} <= names
        assert not any(name.endswith(".pt") or "cache" in name or "previous" in name for name in names)
        assert packed.testzip() is None


def test_blind_instructions_explain_rating_without_exposing_conditions(report, run_dir):
    report.finish_report(run_dir)
    text = (run_dir / "task_blind_instructions.md").read_text(encoding="utf-8")
    assert all(anchor in text for anchor in ("0 分", "50 分", "100 分", "Spearman ρ ≥ 0.6"))
    assert "不要评价回答的人格或语气" in text
    assert "原始向量" not in text and "仅配对" not in text and "cohfilt" not in text


@pytest.mark.parametrize("all_identical", [False, True])
def test_blind_selection_avoids_repeated_qa_when_possible(report, run_dir, all_identical):
    for source in (run_dir / "steps").glob("*/calibration_judge_samples.jsonl"):
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        for record in records:
            for i, sample in enumerate(record["result"]["samples"]):
                if all_identical:
                    sample["q"], sample["a"] = "同一问题", "同一回答"
                elif record["coef"] == 0:
                    sample["a"] = f"跨方法相同的基线回答{i}"
        source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    report.finish_report(run_dir)
    key = json.loads((run_dir / "task_blind_key.json").read_text(encoding="utf-8"))
    with (run_dir / "task_blind.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n_unique = len({(r["question"], r["answer"]) for r in rows})
    assert n_unique == (1 if all_identical else 30)
    assert key["n_unique_qa"] == n_unique and key["n_duplicate_qa"] == 30 - n_unique


def test_rerun_preserves_completed_human_scores_and_row_order(report, run_dir):
    report.finish_report(run_dir)
    path = run_dir / "task_blind.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.reverse()
    for i, row in enumerate(rows):
        row["task_human"] = str(i + 50)
        row["notes"] = "人工备注"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "task_human", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    archive = report.finish_report(run_dir)
    assert path.read_bytes() == before
    assert not (run_dir / "previous_blind").exists()
    with zipfile.ZipFile(archive) as packed:
        assert packed.read("task_blind.csv") == before


@pytest.mark.parametrize("changed", ["question", "missing_key"])
def test_blind_identity_mismatch_backs_up_human_work(report, run_dir, changed):
    report.finish_report(run_dir)
    path = run_dir / "task_blind.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["task_human"] = "85"
    if changed == "question":
        rows[0]["question"] = "不同的问题"
    else:
        (run_dir / "task_blind_key.json").unlink()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    archive = report.finish_report(run_dir)
    assert (run_dir / "previous_blind" / "0001" / "task_blind.csv").read_bytes() == before
    assert path.read_bytes() != before
    assert (run_dir / "previous_blind" / "0001" / "task_blind_key.json").exists() == (changed == "question")
    with zipfile.ZipFile(archive) as packed:
        assert not any("previous_blind" in name for name in packed.namelist())


@pytest.mark.parametrize("mutation", ["missing_condition", "bad_finish_reason", "missing_task"])
def test_incomplete_inputs_do_not_make_result_zip(report, run_dir, mutation):
    source = run_dir / "steps" / "original_c_high" / "calibration_judge_samples.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    if mutation == "missing_condition":
        records.pop()
    elif mutation == "bad_finish_reason":
        records[0]["result"]["samples"][0]["finish_reason"] = "unknown"
    else:
        del records[0]["result"]["samples"][0]["task"]
    source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ValueError):
        report.finish_report(run_dir)
    assert not (run_dir / "results.zip").exists()
