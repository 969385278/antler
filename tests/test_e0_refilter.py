"""C18：旧回答的顺序还原、配对过滤和只前向重算；CPU 离线运行。"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from transformers import BatchEncoding

from test_core import model  # 复用随机微型模型，不下载权重
from antler.traits.data import POLES, load_trait
from antler.traits import vectors


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("e0_refilter")


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_records(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def original_order(pole, n_rollouts, max_questions):
    """独立复写旧版 extract_pole 的循环，避免以被测函数生成期望值。"""
    data = load_trait(pole, "extract")
    result = []
    for pair_index, pair in enumerate(data["instruction"]):
        for question_index, q in enumerate(data["questions"][:max_questions]):
            for rollout in range(n_rollouts):
                result.append(("pos", pair["pos"], q, pair_index, question_index, rollout))
                result.append(("neg", pair["neg"], q, pair_index, question_index, rollout))
    return result


def test_reconstruction_matches_original_extraction_loop(monkeypatch):
    pole = "careless"
    expected = original_order(pole, 2, 3)
    captured = []

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            captured.append((messages[0]["content"], messages[1]["content"]))
            return "提示"

    monkeypatch.setattr(vectors, "generate", lambda *a, **kw: ["旧回答"] * len(expected))
    monkeypatch.setattr(vectors, "response_mean_acts", lambda m, t, ps, rs, **kw: torch.ones(len(ps), 4, 32))
    result = vectors.extract_pole(None, Tok(), pole, n_rollouts=2, max_questions=3)
    restored = vectors.extraction_rows(pole, n_rollouts=2, max_questions=3, samples=result["samples"])
    assert captured == [(system, q) for _, system, q, *_ in expected]
    assert [(r["side"], r["system"], r["question"], r["pair_index"], r["question_index"], r["rollout"])
            for r in restored] == expected
    assert [r["source_index"] for r in restored] == list(range(len(expected)))
    assert result["n_pos"] == result["n_neg"] == len(expected) // 2
    assert vectors.extraction_rows(pole, n_rollouts=1, max_questions=0) == vectors.extraction_rows(pole, n_rollouts=1)


@pytest.mark.parametrize("damage", ["count", "order", "question"])
def test_reconstruction_rejects_mismatched_sources(damage):
    samples = [{"side": side, "question": q} for side, _, q, *_ in original_order("calm", 2, 2)]
    if damage == "count":
        samples.pop()
    elif damage == "order":
        samples[0], samples[1] = samples[1], samples[0]
    else:
        samples[0]["question"] = "错误题目"
    with pytest.raises(ValueError):
        vectors.extraction_rows("calm", n_rollouts=2, max_questions=2, samples=samples)


def test_pair_boundaries_missing_scores_and_trait_only_control():
    scores = [(50, 49), (49, 0), (90, 50), (90, 0), (90, 0), (None, 0)]
    rows = [{"side": side, "question": str(i), "score": score}
            for i, pair in enumerate(scores) for side, score in zip(("pos", "neg"), pair)]
    coherence = [50, 50, 90, 90, 90, 90, 49, 90, None, 90, 90, 90]
    assert vectors.select_paired_samples(rows, coherence, min_coherence=50) == ([0], [1])
    assert vectors.select_paired_samples(rows, coherence, min_coherence=None) == ([0, 6, 8], [1, 7, 9])
    assert vectors.select_paired_samples(rows) == ([0, 6, 8], [1, 7, 9])
    for score in (float("nan"), True, 101):
        coherence[0] = score
        with pytest.raises(ValueError, match="连贯性"):
            vectors.select_paired_samples(rows, coherence, min_coherence=50)


def test_extract_default_keeps_independent_legacy_filter(monkeypatch):
    expected = original_order("dependable", 1, 2)
    scores = [90, 10] * 10
    scores[0] = 50
    scores[3] = 70
    coherence = [90] * 20
    coherence[5] = 49

    class Judge:
        def trait_scores(self, *args):
            return scores

        def coherence_scores(self, *args):
            return coherence

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            return "提示"

    monkeypatch.setattr(vectors, "generate", lambda *a, **kw: ["旧回答"] * len(expected))
    monkeypatch.setattr(vectors, "response_mean_acts", lambda m, t, ps, rs, **kw: torch.ones(len(ps), 4, 32))
    old = vectors.extract_pole(None, Tok(), "dependable", judge=Judge(), n_rollouts=1, max_questions=2)
    assert old["n_pos"] == old["n_neg"] == 9
    assert "coherence" not in old
    new = vectors.extract_pole(None, Tok(), "dependable", judge=Judge(), n_rollouts=1, max_questions=2,
                               min_coherence=50)
    assert new["n_pos"] == new["n_neg"] == 8
    assert new["coherence"] == coherence and new["min_coherence"] == 50
    with pytest.raises(ValueError, match="评委"):
        vectors.extract_pole(None, Tok(), "dependable", min_coherence=50)


@pytest.fixture
def source_data(script, tmp_path):
    out = tmp_path / "main"
    (out / "vectors").mkdir(parents=True)
    rng = torch.Generator().manual_seed(3)
    pole_vectors = {p: torch.randn(4, 32, generator=rng) for p in POLES}
    for pole, vec in pole_vectors.items():
        torch.save(vec, out / "vectors" / f"{pole}.pt")
    directions = {"U": vectors.build_directions(pole_vectors), "s": torch.arange(1, 5).float(),
                  "dims": list(vectors.DIMS), "meta": {"model": "offline-model", "note": "保留来源元数据"},
                  "custom": "保留扩展字段"}
    torch.save(directions, out / "directions.pt")
    records = []
    for pole in script.DEFAULT_POLES:
        samples = [{"side": side, "question": q, "answer": f"旧回答{index}",
                    "score": 90 if side == "pos" else 10}
                   for index, (side, _, q, *_) in enumerate(original_order(pole, 1, 2))]
        path = out / "vectors" / f"{pole}_samples.json"
        path.write_bytes(json.dumps(samples, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode())
        digest = script.sha256(path)
        for i, sample in enumerate(samples):
            records.append({"pole": pole, "source_index": i, "source_sha256": digest,
                            "source_file": f"/another-machine/vectors/{pole}_samples.json",
                            "side": sample["side"], "question": sample["question"],
                            "answer": sample["answer"], "trait_score": sample["score"],
                            "coherence": 50 if i == 0 else 90})
    write_records(out / "extract_coherence_samples.jsonl", records)
    config = tmp_path / "config.yaml"
    config.write_text(f"model: offline-model\nout_dir: {out.as_posix()}\nn_rollouts: 1\n"
                      "max_questions: 2\nbatch_size: 20\ndtype: float32\ndevice: cpu\n", encoding="utf-8")
    cfg = {"model": "offline-model", "out_path": out, "n_rollouts": 1, "max_questions": 2,
           "batch_size": 20, "dtype": "float32", "device": "cpu"}
    parser = argparse.ArgumentParser()
    script.add_arguments(parser)
    args = parser.parse_args([])
    args.config = str(config)
    return args, cfg, records, directions


def test_prepare_inputs_counts_cross_machine_paths_and_hashes(script, source_data):
    _, cfg, records, _ = source_data
    records[3]["coherence"] = None
    write_records(cfg["out_path"] / "extract_coherence_samples.jsonl", records)
    result = script.prepare_inputs(cfg, cfg["out_path"], script.DEFAULT_POLES)
    counts = result["inputs"]["dependable"]["counts"]
    assert counts["n_kept_pairs"] == 9 and counts["n_trait_pairs"] == 10
    assert counts["n_missing_coherence"] == counts["n_trait_pairs_missing_coherence"] == 1
    paired = script.prepare_inputs(cfg, cfg["out_path"], script.DEFAULT_POLES, paired_only=True)
    assert paired["inputs"]["dependable"]["counts"]["n_kept_pairs"] == 10
    assert set(result["vectors"]) == set(POLES)
    assert str(cfg["out_path"] / "directions.pt") in result["data_hashes"]
    assert len([p for p in result["data_hashes"] if "trait_extract" in p]) == 4


@pytest.mark.parametrize("damage", ["hash", "index", "duplicate", "missing", "question", "answer",
                                    "side", "trait_score", "score_bool", "missing_pole", "empty",
                                    "shape", "directions", "scale", "too_few", "wrong_model"])
def test_preflight_rejects_before_loading_model(script, source_data, monkeypatch, damage):
    args, cfg, records, directions = source_data
    out = cfg["out_path"]
    if damage == "hash":
        records[0]["source_sha256"] = "错误来源哈希"
    elif damage == "index":
        records[0]["source_index"] = True
    elif damage == "duplicate":
        records.append(records[0])
    elif damage == "missing":
        records.pop()
    elif damage in ("question", "answer", "side", "trait_score"):
        records[0][damage] = "与源文件不同"
    elif damage == "score_bool":
        records[0]["coherence"] = True
    elif damage == "missing_pole":
        (out / "vectors" / "inventive.pt").unlink()
    elif damage == "empty":
        path = out / "vectors" / "dependable_samples.json"
        samples = json_read(path)
        samples[0]["answer"] = ""
        json_write(path, samples)
        for record in records[:20]:
            record["source_sha256"] = script.sha256(path)
        records[0]["answer"] = ""
    elif damage == "shape":
        torch.save(torch.zeros(2, 32), out / "vectors" / "inventive.pt")
    elif damage == "directions":
        directions["U"][0, 0, 0] += 0.2
        torch.save(directions, out / "directions.pt")
    elif damage == "scale":
        directions["s"][0] = 0
        torch.save(directions, out / "directions.pt")
    elif damage == "too_few":
        for record in records[:12]:
            record["coherence"] = 49
    elif damage == "wrong_model":
        directions["meta"]["model"] = "different-model"
        torch.save(directions, out / "directions.pt")
    write_records(out / "extract_coherence_samples.jsonl", records)
    monkeypatch.setattr(script, "load_model", lambda *a, **kw: pytest.fail("预检失败不应加载模型"))
    with pytest.raises((ValueError, FileNotFoundError)):
        script.run_refilter(args)
    assert not out.with_name("main_cohfilt").exists()


class TinyTokenizer:
    """为微型模型构造短 token 输入，避免使用联网分词器。"""
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False) + "<回答开始>"

    def __call__(self, text, return_tensors=None, **kwargs):
        def ids(value):
            prompt, _, answer = value.partition("<回答开始>")
            prefix = int(hashlib.sha256(prompt.encode()).hexdigest()[:4], 16) % 60 + 1
            result = [1, prefix, 2]
            if answer:
                result += [int(hashlib.sha256(answer.encode()).hexdigest()[:4], 16) % 60 + 1, 3]
            return result
        if isinstance(text, str):
            return {"input_ids": ids(text)}
        batch = [ids(value) for value in text]
        assert len({len(row) for row in batch}) == 1
        return BatchEncoding({"input_ids": torch.tensor(batch), "attention_mask": torch.ones(len(batch), len(batch[0]), dtype=torch.long)})


@pytest.mark.parametrize("paired_only", [False, True])
def test_real_forward_outputs_frozen_model_and_preserves_sources(script, source_data, model, monkeypatch, paired_only):
    args, cfg, records, original = source_data
    args.paired_only = paired_only
    # 测试明确指定为原目录下的新运行子目录。
    args.output_dir = str(cfg["out_path"] / "diagnosis" / ("paired" if paired_only else "cohfilt"))
    records[3]["coherence"] = None
    write_records(cfg["out_path"] / "extract_coherence_samples.jsonl", records)
    sources = {p: p.read_bytes() for p in cfg["out_path"].rglob("*") if p.is_file()}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    calls = []

    def load(*args):
        calls.append(args)
        return model, TinyTokenizer()

    monkeypatch.setattr(script, "load_model", load)
    monkeypatch.setattr(model, "generate", lambda *a, **kw: pytest.fail("禁止生成新回答"))
    judge_module = importlib.import_module("antler.eval.judge")
    monkeypatch.setattr(judge_module, "Judge", lambda *a, **kw: pytest.fail("禁止构造评委"))
    result_dir = script.run_refilter(args)
    assert len(calls) == 1
    assert all(p.read_bytes() == raw for p, raw in sources.items())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not layer._forward_hooks for layer in model.model.layers)
    result = torch.load(result_dir / "directions.pt", weights_only=True)
    assert torch.equal(result["s"], original["s"]) and result["dims"] == original["dims"]
    assert result["meta"]["note"] == "保留来源元数据" and result["custom"] == original["custom"]
    assert result["U"].shape == original["U"].shape
    assert torch.allclose(result["U"].norm(dim=-1), torch.ones(4, 5), atol=1e-5)
    for pole in POLES:
        output_vec = result_dir / "vectors" / f"{pole}.pt"
        assert output_vec.is_file()
        if pole not in script.DEFAULT_POLES:
            assert output_vec.read_bytes() == sources[cfg["out_path"] / "vectors" / f"{pole}.pt"]
    summary = json_read(result_dir / "refilter_summary.json")
    assert summary["poles"]["dependable"]["n_kept_pairs"] == (10 if paired_only else 9)
    assert len(summary["cn_cosine_before"]) == len(summary["cn_cosine_after"]) == 4
    meta = json_read(result_dir / "run_meta.json")
    assert meta["status"] == "completed" and meta["task"] == "C18" and meta["end_time"]
    assert meta["model_loaded"] and not meta["generation_performed"] and not meta["judge_called"]
    assert meta["inferred_reconstruction"] and meta["reconstruction_limitation"]
    assert meta["min_coherence"] == (None if paired_only else 50)
    assert len(meta["copied_poles"]) == 6
    assert meta["config_sha256"] and meta["data_hashes"]
    for relative, digest in meta["artifact_hashes"].items():
        assert script.sha256(result_dir / relative) == digest
    selection = [json.loads(line) for line in (result_dir / "refilter_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(selection) == 80 and selection[0]["system"]
    assert selection[0]["kept"]
    assert selection[3]["kept"] == paired_only


def test_nonempty_output_never_overwritten(script, source_data, monkeypatch):
    args, cfg, _, _ = source_data
    output = cfg["out_path"].with_name("main_cohfilt")
    output.mkdir()
    path = output / "run_meta.json"
    path.write_text('{"status":"failed"}', encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(script, "load_model", lambda *a, **kw: pytest.fail("不得覆盖旧目录"))
    with pytest.raises(FileExistsError):
        script.run_refilter(args)
    assert path.read_bytes() == before


@pytest.mark.parametrize("target", ["source", "parent", "vectors", "vectors_child"])
def test_rejects_output_that_can_overwrite_sources(script, source_data, target):
    args, cfg, _, _ = source_data
    source = cfg["out_path"]
    args.output_dir = str({"source": source, "parent": source.parent,
                           "vectors": source / "vectors", "vectors_child": source / "vectors" / "new"}[target])
    with pytest.raises(ValueError, match="不能覆盖"):
        script.run_refilter(args)


def test_forward_failure_records_failed_without_changing_source(script, source_data, model, monkeypatch):
    args, cfg, _, _ = source_data
    before = {path: path.read_bytes() for path in cfg["out_path"].rglob("*") if path.is_file()}
    monkeypatch.setattr(script, "load_model", lambda *a, **kw: (model, TinyTokenizer()))

    def fail(*a, **kw):
        raise RuntimeError("模拟前向失败，带有敏感信息")

    monkeypatch.setattr(script, "recompute_pole", fail)
    with pytest.raises(RuntimeError, match="模拟前向失败"):
        script.run_refilter(args)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    output = cfg["out_path"].with_name("main_cohfilt")
    raw = (output / "run_meta.json").read_text(encoding="utf-8")
    assert "敏感信息" not in raw
    meta = json.loads(raw)
    assert meta["status"] == "failed" and meta["error_type"] == "RuntimeError" and meta["end_time"]
    assert not (output / "directions.pt").exists()
