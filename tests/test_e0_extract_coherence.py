"""C17：已有回答补打分的离线测试；禁止加载模型、生成回答或访问评委 API。"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("e0_extract_coherence")


def arguments(script, *argv):
    ap = argparse.ArgumentParser()
    script.add_arguments(ap)
    return ap.parse_args(argv)


def write_source(out_path, pole, rows=None):
    if rows is None:
        rows = [{"side": "pos", "question": "同一个问题", "answer": "正向回答", "score": 90},
                {"side": "neg", "question": "同一个问题", "answer": "负向回答", "score": None},
                {"side": "pos", "question": "同一个问题", "answer": "另一次正向回答", "score": 55},
                {"side": "neg", "question": "同一个问题", "answer": "另一次负向回答", "score": 10}]
    path = out_path / "vectors" / f"{pole}_samples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # 刻意使用 CRLF，验证脚本不会重写源文件或破坏其哈希。
    raw = json.dumps(rows, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8")
    path.write_bytes(raw)
    return path, raw


def runtime(script, monkeypatch, tmp_path, scores):
    """使用真实公共 setup，仅替换模型加载器和评委，任何加载尝试都报错。"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(f"model: unavailable/model\nout_dir: {tmp_path.as_posix()}\nuse_judge: true\n",
                           encoding="utf-8")
    common = importlib.import_module("_common")
    judge_module = importlib.import_module("antler.eval.judge")

    def forbidden(*args, **kwargs):
        pytest.fail("C17 不得加载或下载模型")

    monkeypatch.setattr(common, "load_model", forbidden)
    calls = []

    class FakeJudge:
        model = "固定测试评委"
        base_url = "https://example.invalid/v1"
        max_tokens = 16
        extra_body = {"thinking": {"type": "disabled"}}

        def __init__(self, cache_path):
            self.cache_path = cache_path

        def coherence_scores(self, questions, answers):
            calls.append((questions, answers))
            result = scores[len(calls) - 1]
            if isinstance(result, BaseException):
                raise result
            return result

        def trait_scores(self, *args, **kwargs):
            pytest.fail("C17 不得重打人格分")

    monkeypatch.setattr(judge_module, "Judge", FakeJudge)
    monkeypatch.setattr(sys, "argv", ["e0_extract_coherence.py", "--config", str(config_path)])
    return calls


def test_cli_defaults_and_all_poles(script):
    assert arguments(script).poles == ["dependable", "careless", "nervous", "calm"]
    assert arguments(script, "--poles", *script.POLES).poles == script.POLES
    for argv in (["--poles"], ["--poles", "unknown"]):
        with pytest.raises(SystemExit):
            arguments(script, *argv)
    with pytest.raises(ValueError, match="不重复"):
        script.load_inputs(Path("unused"), ["calm", "calm"])


def test_summary_valid_denominator_and_strict_boundary(script):
    assert script.summarize([None, 0, 49, 50, 100]) == {
        "n_total": 5, "n_valid": 4, "n_missing": 1, "n_below_threshold": 2,
        "mean_coherence": 49.75, "below_threshold_fraction": 0.5}
    for scores in ([], [None, None]):
        group = script.summarize(scores)
        assert group["n_total"] == len(scores) and group["n_missing"] == len(scores)
        assert group["n_valid"] == 0 and group["n_below_threshold"] == 0
        assert group["mean_coherence"] is None and group["below_threshold_fraction"] is None


def test_default_run_outputs_and_source_alignment(script, monkeypatch, tmp_path):
    sources = {pole: write_source(tmp_path, pole) for pole in script.DEFAULT_POLES}
    calls = runtime(script, monkeypatch, tmp_path, [[49, None, 50, None]] * 4)
    script.main()
    assert len(calls) == 4
    assert calls[0] == (["同一个问题"] * 4, ["正向回答", "负向回答", "另一次正向回答", "另一次负向回答"])
    summary = json.loads((tmp_path / "extract_coherence.json").read_text(encoding="utf-8"))
    assert summary["threshold"] == 50 and summary["below_threshold_denominator"] == "n_valid"
    assert list(summary["poles"]) == script.DEFAULT_POLES
    records = [json.loads(line) for line in (tmp_path / "extract_coherence_samples.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    assert len(records) == 16
    for index, pole in enumerate(script.DEFAULT_POLES):
        path, original = sources[pole]
        assert path.read_bytes() == original
        rows = records[index * 4:(index + 1) * 4]
        assert [r["source_index"] for r in rows] == list(range(4))
        assert [r["side"] for r in rows] == ["pos", "neg", "pos", "neg"]
        assert [r["trait_score"] for r in rows] == [90, None, 55, 10]
        assert [r["coherence"] for r in rows] == [49, None, 50, None]
        assert all(r["pole"] == pole and r["source_file"] == str(path.resolve()) for r in rows)
        assert all(r["source_sha256"] == hashlib.sha256(original).hexdigest() for r in rows)
        group = summary["poles"][pole]
        assert group["pos"]["mean_coherence"] == 49.5
        assert group["pos"]["below_threshold_fraction"] == 0.5
        assert group["neg"]["n_missing"] == 2 and group["neg"]["mean_coherence"] is None
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "C17" and meta["status"] == "completed" and meta["end_time"]
    assert not meta["model_loaded"] and not meta["generation_performed"]
    assert meta["poles"] == script.DEFAULT_POLES
    assert len(meta["data_hashes"]) == 4 and meta["config_sha256"]
    assert meta["judge"]["model"] == "固定测试评委"


def test_custom_poles_preserve_requested_order(script, monkeypatch, tmp_path):
    for pole in ("calm", "inventive"):
        write_source(tmp_path, pole)
    calls = runtime(script, monkeypatch, tmp_path, [[70] * 4, [90] * 4])
    monkeypatch.setattr(sys, "argv", sys.argv + ["--poles", "calm", "inventive"])
    script.main()
    assert len(calls) == 2
    summary = json.loads((tmp_path / "extract_coherence.json").read_text(encoding="utf-8"))
    assert list(summary["poles"]) == ["calm", "inventive"]
    assert summary["poles"]["calm"]["pos"]["mean_coherence"] == 70


@pytest.mark.parametrize("bad_input", ["missing", "wrong_side", "not_list", "bad_score"])
def test_all_inputs_validated_before_scoring_or_output(script, monkeypatch, tmp_path, bad_input):
    for pole in script.DEFAULT_POLES[:-1]:
        write_source(tmp_path, pole)
    if bad_input == "wrong_side":
        write_source(tmp_path, "calm", [{"side": "other", "question": "q", "answer": "a", "score": 20}])
    elif bad_input == "not_list":
        write_source(tmp_path, "calm", {"samples": []})
    elif bad_input == "bad_score":
        write_source(tmp_path, "calm", [{"side": "pos", "question": "q", "answer": "a", "score": float("nan")}])
    previous = tmp_path / "extract_coherence.json"
    previous.write_text("已有结果", encoding="utf-8")
    calls = runtime(script, monkeypatch, tmp_path, [])
    with pytest.raises((ValueError, FileNotFoundError)):
        script.main()
    assert calls == []
    assert previous.read_text(encoding="utf-8") == "已有结果"
    assert not (tmp_path / "run_meta.json").exists()
    assert not (tmp_path / "extract_coherence_samples.jsonl").exists()


def test_failed_scoring_preserves_results_and_records_failure(script, monkeypatch, tmp_path):
    sources = [write_source(tmp_path, pole) for pole in script.DEFAULT_POLES]
    for name in ("extract_coherence.json", "extract_coherence_samples.jsonl"):
        (tmp_path / name).write_text("已有结果", encoding="utf-8")
    calls = runtime(script, monkeypatch, tmp_path, [[90] * 4, RuntimeError("模拟中断和敏感令牌")])
    with pytest.raises(RuntimeError, match="模拟中断"):
        script.main()
    assert len(calls) == 2
    for name in ("extract_coherence.json", "extract_coherence_samples.jsonl"):
        assert (tmp_path / name).read_text(encoding="utf-8") == "已有结果"
    for path, raw in sources:
        assert path.read_bytes() == raw
    raw_meta = (tmp_path / "run_meta.json").read_text(encoding="utf-8")
    meta = json.loads(raw_meta)
    assert meta["status"] == "failed" and meta["end_time"]
    assert meta["error_type"] == "RuntimeError" and "敏感令牌" not in raw_meta


def test_metadata_removes_credentials(script, tmp_path, monkeypatch):
    args = argparse.Namespace(config="not_present.yaml")
    cfg = {"out_path": tmp_path, "api_key": "配置凭据", "extra": {"Authorization": "配置授权"}}
    judge = argparse.Namespace(model="评委", api_key="评委凭据", max_tokens=16,
                               base_url="https://user:网址密码@example.invalid/v1?token=查询凭据",
                               extra_body={"headers": {"authorization": "请求授权", "api-key": "请求凭据"}})
    monkeypatch.setattr(sys, "argv", ["e0_extract_coherence.py"])
    with script.record_run(args, cfg, judge, {}):
        pass
    raw = (tmp_path / "run_meta.json").read_text(encoding="utf-8")
    for secret in ("配置凭据", "配置授权", "评委凭据", "网址密码", "查询凭据", "请求授权", "请求凭据"):
        assert secret not in raw
    assert json.loads(raw)["judge"]["base_url"] == "https://example.invalid/v1"


def test_setup_default_still_loads_model(script, tmp_path, monkeypatch):
    common = importlib.import_module("_common")
    cfg = {"out_path": tmp_path, "model": "本地模型", "dtype": "float32", "device": "cpu"}
    monkeypatch.setattr(common, "load_config", lambda _: cfg)
    calls = []
    sentinel_model, sentinel_tokenizer = object(), object()

    def load(*args):
        calls.append(args)
        return sentinel_model, sentinel_tokenizer

    monkeypatch.setattr(common, "load_model", load)
    args = argparse.Namespace(config="测试配置")
    assert common.setup(args, need_judge=False) == (cfg, sentinel_model, sentinel_tokenizer, None)
    assert calls == [("本地模型", "float32", "cpu")]
    assert common.setup(args, need_judge=False, need_model=False) == (cfg, None, None, None)
    assert len(calls) == 1
