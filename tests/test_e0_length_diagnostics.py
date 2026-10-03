"""长度诊断离线测试：原始token切片、闸门和独立结果，不调用模型下载或真实评委。"""
import importlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class Tok:
    eos_token_id = 2
    pad_token_id = 0

    def batch_decode(self, rows, **kwargs):
        return [" ".join(str(x) for x in row if x != 2) for row in rows]


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("e0_length_diagnostics")


def item(tokens, reason):
    return {"token_ids": tokens, "text": Tok().batch_decode([tokens])[0],
            "n_generated_tokens": len(tokens), "finish_reason": reason}


def test_views_slice_original_tokens_and_keep_boundary_eos(script):
    r = item([7] * 319 + [2], "eos")
    views = script.token_views(r, Tok(), [2], 1024)
    assert views["prefix320"] == views["long"]
    r = item([7] * 330 + [2], "eos")
    views = script.token_views(r, Tok(), [2], 1024)
    assert views["prefix320"]["text"] == " ".join(["7"] * 320)
    assert views["prefix320"]["finish_reason"] == "length"
    assert views["long"]["finish_reason"] == "eos"


@pytest.mark.parametrize("change", ["count", "reason", "text", "early_eos", "short_length"])
def test_mismatched_raw_tokens_rejected(script, change):
    r = item([7] * 330 + [2], "eos")
    if change == "count": r["n_generated_tokens"] -= 1
    if change == "reason": r["finish_reason"] = "length"
    if change == "text": r["text"] = "错误文本"
    if change == "early_eos": r["token_ids"][4] = 2
    if change == "short_length": r = item([7] * 330, "length")
    with pytest.raises(ValueError):
        script.token_views(r, Tok(), [2], 1024)


def test_same_view_scores_once_and_missing_scores_stay_missing(script):
    calls = []
    judge = SimpleNamespace(trait_scores=lambda *a: calls.append("trait") or [None],
                            coherence_scores=lambda *a: calls.append("coherence") or [80],
                            task_scores=lambda *a: calls.append("task") or [90])
    views = {name: {"text": "同一回答"} for name in ("prefix320", "long")}
    script.score_views(judge, "题目", "提示", views)
    assert calls == ["trait", "coherence", "task"]
    assert views["long"]["scores"]["trait"] is None


@pytest.mark.parametrize("natural_eos", [True, False])
def test_full_flow_preserves_source_and_gates_judge(script, tmp_path, monkeypatch, natural_eos):
    source = tmp_path / "main"
    parent = source / "refilter_diagnostics" / "old_run"
    d = {"U": torch.randn(4, 5, 32), "s": torch.ones(4), "dims": script.DIMS}
    paths = {}
    for variant in ("original", "cohfilt"):
        p = parent / "variants" / variant / "directions.pt"
        p.parent.mkdir(parents=True)
        torch.save(d, p)
        paths[variant] = p
    meta = {"status": "completed", "provenance": {
        "file_hashes": {str(source / "directions.pt"): script.sha256(paths["original"])},
        "judge": {"model": "offline-judge"}}, "steps": {"refilter_cohfilt": {
            "artifact_hashes": {"variants/cohfilt/directions.pt": script.sha256(paths["cohfilt"])}}}}
    script.write_json(parent / "run_meta.json", meta)
    config = tmp_path / "config.yaml"; config.write_text("model: offline\n", encoding="utf-8")
    cfg = {"out_path": source, "model": "offline", "device": "cpu", "dtype": "float32"}
    args = SimpleNamespace(config=str(config), refilter_run=str(parent), long_tokens=1024, layer=1)
    monkeypatch.setattr(script, "setup", lambda *a, **k: (cfg, None, None, None))
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=4, hidden_size=32, eos_token_id=2),
                            generation_config=SimpleNamespace(eos_token_id=[2], to_dict=lambda: {"eos_token_id": [2]}))
    monkeypatch.setattr(script, "load_model", lambda *a: (model, Tok()))
    monkeypatch.setattr(script, "chat_prompt", lambda tok, q: q)
    generations = []
    def fake_generate(*a, **kwargs):
        generations.append(kwargs)
        return [item([7]*330+[2], "eos") if natural_eos else item([7]*1024, "length")]
    monkeypatch.setattr(script, "generate", fake_generate)
    judge_calls = []
    def fake_judge(**kwargs):
        judge_calls.append(kwargs)
        return SimpleNamespace(model="offline-judge", base_url="offline", max_tokens=16, extra_body={},
                               trait_scores=lambda *a: [50], coherence_scores=lambda *a: [80], task_scores=lambda *a: [None])
    monkeypatch.setattr(script, "Judge", fake_judge)
    before = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    out = script.run_diagnostic(args)
    result = script.read_json(out / "run_meta.json")
    assert result["status"] == ("completed" if natural_eos else "stopped_no_baseline_eos")
    assert len(generations) == (12 if natural_eos else 4)
    assert len(judge_calls) == int(natural_eos)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert all(g["return_token_ids"] and g["seed"] == 0 and g["batch_size"] == 1 for g in generations)
    rows = [json.loads(l) for l in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(generations)
    if natural_eos:
        assert rows[0]["views"]["long"]["scores"]["task"] is None
        assert sum(g["steer"] is None for g in generations) == 4
    with zipfile.ZipFile(out / "results.zip") as bundle:
        assert bundle.testzip() is None and "samples.jsonl" in bundle.namelist()
        assert not any("cache" in n or n.endswith(".pt") for n in bundle.namelist())
