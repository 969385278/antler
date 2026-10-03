"""C18：读取旧回答重算配对过滤向量；不生成回答、不调用评委、不覆盖原结果。"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.activations import response_mean_acts
from antler.core.model import chat_prompt, freeze, load_model
from antler.traits.data import DIMS, KIT, POLES
from antler.traits.vectors import build_directions, extraction_rows, select_paired_samples
from e0_extract_coherence import DEFAULT_POLES, _redacted, _score_is_valid, _write_json, load_inputs


def add_arguments(ap):
    ap.add_argument("--source-out", help="原始提取与 C17 产出目录，默认配置输出目录")
    ap.add_argument("--output-dir", help="新向量目录，默认原目录同级的 <名称>_cohfilt")
    ap.add_argument("--poles", nargs="+", choices=POLES, default=DEFAULT_POLES)
    ap.add_argument("--min-coherence", type=float, default=50,
                    help="两侧回答均需达到的连贯性下限，默认 50")
    ap.add_argument("--paired-only", action="store_true",
                    help="仅配对筛选人格分，不筛连贯性；默认输出到 <名称>_paired，作为匹配对照")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs(cfg, source_out, poles, min_coherence=50, paired_only=False):
    """在加载模型前校验全部旧回答、C17 对齐、向量及至少五对可用样本。"""
    source_out = Path(source_out).resolve()
    select_paired_samples([], [], min_coherence=min_coherence)
    inputs = load_inputs(source_out, poles)
    directions_path = source_out / "directions.pt"
    directions = torch.load(directions_path, map_location="cpu", weights_only=True)
    if not isinstance(directions, dict) or directions.get("dims") != DIMS:
        raise ValueError("原 directions.pt 的维度顺序必须是 O C E A N")
    U, s = directions.get("U"), directions.get("s")
    if (not isinstance(U, torch.Tensor) or U.ndim != 3 or U.shape[1] != len(DIMS)
            or U.shape[0] < 1 or U.shape[2] < 1 or not U.is_floating_point()
            or not torch.isfinite(U).all()):
        raise ValueError("原 directions.pt 的 U 形状或数值无效")
    if not torch.allclose(U.norm(dim=-1), torch.ones_like(U[..., 0]), atol=1e-5):
        raise ValueError("原 directions.pt 的方向必须为非零单位向量")
    if (not isinstance(s, torch.Tensor) or s.shape != (U.shape[0],)
            or not torch.isfinite(s).all() or not (s > 0).all()):
        raise ValueError("原 directions.pt 缺少有效的层尺度 s")
    if not isinstance(directions.get("meta", {}), dict):
        raise ValueError("原 directions.pt 的 meta 必须是字典")
    source_model = directions.get("meta", {}).get("model")
    if source_model is not None and source_model != cfg["model"]:
        raise ValueError("配置模型与原 directions.pt 的模型不一致，不能混用激活")
    vectors = {}
    hashes = {str(directions_path): sha256(directions_path)}
    for pole in POLES:
        path = source_out / "vectors" / f"{pole}.pt"
        vec = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(vec, torch.Tensor) or vec.shape != (U.shape[0], U.shape[2])
                or not vec.is_floating_point() or not torch.isfinite(vec).all()):
            raise ValueError(f"{pole}: 来源向量形状或数值无效")
        vectors[pole] = vec
        hashes[str(path)] = sha256(path)
        samples_path = source_out / "vectors" / f"{pole}_samples.json"
        if samples_path.is_file():
            hashes[str(samples_path)] = sha256(samples_path)
    rebuilt = build_directions(vectors)
    if not torch.allclose(rebuilt.to(U.dtype), U, atol=1e-5, rtol=1e-4):
        raise ValueError("原 directions.pt 与十个极点向量不一致")
    coherence_path = source_out / "extract_coherence_samples.jsonl"
    by_pole = {pole: {} for pole in poles}
    for line_number, line in enumerate(coherence_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"C17 第 {line_number} 行不是有效记录")
        pole = record.get("pole")
        if pole not in by_pole:
            continue
        index = record.get("source_index")
        rows = inputs[pole]["samples"]
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows)
                or index in by_pole[pole]):
            raise ValueError(f"{pole}: C17 source_index 越界或重复")
        if record.get("source_sha256") != inputs[pole]["source_sha256"]:
            raise ValueError(f"{pole}: C17 source_sha256 与当前回答文件不一致")
        # source_file 可能来自另一台机器；用内容哈希和逐字段检查确定来源。
        for field, source_field in (("side", "side"), ("question", "question"),
                                    ("answer", "answer"), ("trait_score", "score")):
            if field not in record or record[field] != rows[index][source_field]:
                raise ValueError(f"{pole}: C17 source_index={index} 的 {field} 与原回答不一致")
        if record["trait_score"] is not None and not _score_is_valid(record["trait_score"]):
            raise ValueError(f"{pole}: C17 人格分无效")
        if "coherence" not in record or (record["coherence"] is not None
                                          and not _score_is_valid(record["coherence"])):
            raise ValueError(f"{pole}: C17 连贯性分无效")
        by_pole[pole][index] = record["coherence"]
    hashes[str(coherence_path)] = sha256(coherence_path)
    for pole, source in inputs.items():
        samples = source["samples"]
        rows = extraction_rows(pole, n_rollouts=cfg.get("n_rollouts", 2),
                               max_questions=cfg.get("max_questions"), samples=samples)
        if len(by_pole[pole]) != len(samples):
            raise ValueError(f"{pole}: C17 连贯性记录不完整")
        coherence = [by_pole[pole][i] for i in range(len(samples))]
        trait_pos, _ = select_paired_samples(samples)
        pos, neg = select_paired_samples(samples, coherence,
                                         min_coherence=None if paired_only else min_coherence)
        if len(pos) < 5:
            raise ValueError(f"{pole}: 过滤后样本太少，仅保留 {len(pos)} 对，至少需要 5 对")
        if any(not samples[i]["answer"].strip() for i in pos + neg):
            raise ValueError(f"{pole}: 保留样本含空回答，不能提取回答激活")
        source.update(rows=rows, coherence=coherence, pos_indices=pos, neg_indices=neg,
                      counts={"n_total": len(samples), "n_pairs": len(samples) // 2,
                              "n_trait_pairs": len(trait_pos), "n_kept_pairs": len(pos),
                              "n_missing_coherence": sum(v is None for v in coherence),
                              "n_trait_pairs_missing_coherence": sum(
                                  coherence[i] is None or coherence[i + 1] is None for i in trait_pos),
                              "n_filtered_pairs": len(samples) // 2 - len(pos)})
        hashes[source["source_file"]] = source["source_sha256"]
        trait_path = KIT / "trait_extract" / f"{pole}.json"
        hashes[str(trait_path.resolve())] = sha256(trait_path)
    return {"source_out": source_out, "inputs": inputs, "directions": directions,
            "vectors": vectors, "data_hashes": hashes}


def recompute_pole(model, tok, source, *, batch_size):
    """仅对保留的旧回答做前向计算，不改变模型权重。"""
    acts = []
    for indices in (source["pos_indices"], source["neg_indices"]):
        prompts = [chat_prompt(tok, source["rows"][i]["question"], system=source["rows"][i]["system"])
                   for i in indices]
        responses = [source["samples"][i]["answer"] for i in indices]
        group = response_mean_acts(model, tok, prompts, responses, batch_size=max(1, batch_size // 2))
        if not torch.isfinite(group).all():
            raise ValueError("保留回答产生非有限激活，停止重算")
        acts.append(group.mean(0))
    return acts[0] - acts[1]


def _git(*argv):
    try:
        return subprocess.check_output(["git", *argv], cwd=ROOT, text=True, encoding="utf-8",
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_refilter(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    source_out = Path(args.source_out or cfg["out_path"]).resolve()
    suffix = "_paired" if args.paired_only else "_cohfilt"
    output = Path(args.output_dir).resolve() if args.output_dir else source_out.with_name(source_out.name + suffix)
    if (output == source_out or source_out.is_relative_to(output)
            or output == source_out / "vectors" or output.is_relative_to(source_out / "vectors")):
        raise ValueError("新输出目录不能覆盖原目录、其父级或原 vectors 目录")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"输出目录非空，保留已有结果；请改用新目录：{output}")
    prepared = prepare_inputs(cfg, source_out, args.poles, args.min_coherence, args.paired_only)
    batch_size = cfg.get("batch_size", 16)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size 必须是正整数")
    if cfg.get("dtype", "bfloat16") not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype 必须是 float32、float16 或 bfloat16")
    output.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "out_path": output}
    now = lambda: datetime.now(timezone.utc).isoformat()
    dirty = _git("status", "--porcelain")
    meta = {"task": "C18", "experiment": "e0_refilter", "status": "running",
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(dirty) if dirty is not None else None,
            "config_path": str(Path(args.config).resolve()), "config_sha256": sha256(args.config),
            "config": _redacted(cfg), "argv": _redacted(sys.argv[1:]),
            "start_time": now(), "end_time": None, "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "judge": None, "judge_called": False, "generation_performed": False,
            "model_loaded": False, "model_frozen": True, "seed": cfg.get("seed", 0), "model": cfg["model"],
            "source_out": str(source_out), "data_hashes": prepared["data_hashes"],
            "poles": list(args.poles), "paired_only": args.paired_only,
            "min_coherence": None if args.paired_only else args.min_coherence,
            "filter_rule": "成对 pos >= 50，neg < 50" + ("" if args.paired_only else "，两侧 coherence >= 下限"),
            "inferred_reconstruction": True,
            "reconstruction_basis": {"n_rollouts": cfg.get("n_rollouts", 2),
                                     "max_questions": cfg.get("max_questions"),
                                     "order": "pair_index/question_index/rollout/pos-neg"},
            "reconstruction_limitation": "旧样本未保存系统提示及其哈希；按当前提取题库还原并逐条核对 side/question，无法证明历史提示字节完全相同",
            "copied_poles": [p for p in POLES if p not in args.poles],
            "source_metadata": _redacted(prepared["directions"].get("meta", {}))}
    _write_json(output / "run_meta.json", meta)
    try:
        model, tok = load_model(cfg["model"], cfg.get("dtype", "bfloat16"), cfg.get("device", "auto"))
        freeze(model)
        model.eval()
        meta["model_loaded"] = True
        expected = prepared["directions"]["U"].shape
        if (model.config.num_hidden_layers, model.config.hidden_size) != (expected[0], expected[2]):
            raise ValueError("加载模型的层数或隐藏维度与原向量不一致")
        vectors = dict(prepared["vectors"])
        for pole, source in prepared["inputs"].items():
            print(f"[重算] {pole}：保留 {source['counts']['n_kept_pairs']} 对回答")
            vec = recompute_pole(model, tok, source, batch_size=batch_size)
            if vec.shape != vectors[pole].shape or not torch.isfinite(vec).all():
                raise ValueError(f"{pole}: 重算后的向量形状或数值无效")
            vectors[pole] = vec
        U = build_directions(vectors)
        if not torch.isfinite(U).all() or not torch.allclose(U.norm(dim=-1), torch.ones_like(U[..., 0]), atol=1e-5):
            raise ValueError("新方向存在零向量或非有限值")
        for path, digest in prepared["data_hashes"].items():
            if sha256(path) != digest:
                raise ValueError(f"计算期间来源文件发生变化：{path}")
        vdir = output / "vectors"
        vdir.mkdir()
        for pole in POLES:
            if pole in prepared["inputs"]:
                torch.save(vectors[pole], vdir / f"{pole}.pt")
            else:
                shutil.copyfile(source_out / "vectors" / f"{pole}.pt", vdir / f"{pole}.pt")
            samples_path = source_out / "vectors" / f"{pole}_samples.json"
            if samples_path.is_file():
                shutil.copyfile(samples_path, vdir / samples_path.name)
        directions = dict(prepared["directions"])
        directions["U"] = U
        directions["meta"] = {**directions.get("meta", {}), "refilter": {
            "task": "C18", "source_directions_sha256": prepared["data_hashes"][str(source_out / "directions.pt")],
            "poles": list(args.poles), "paired_only": args.paired_only, "min_coherence": meta["min_coherence"]}}
        torch.save(directions, output / "directions.pt")
        original_U = prepared["directions"]["U"]
        ci, ni = DIMS.index("C"), DIMS.index("N")
        summary = {"schema_version": 1, "paired_only": args.paired_only,
                   "min_coherence": meta["min_coherence"], "poles": {
                       p: source["counts"] for p, source in prepared["inputs"].items()},
                   "cn_cosine_before": (original_U[:, ci] * original_U[:, ni]).sum(-1).tolist(),
                   "cn_cosine_after": (U[:, ci] * U[:, ni]).sum(-1).tolist()}
        _write_json(output / "refilter_summary.json", summary)
        with (output / "refilter_samples.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for pole, source in prepared["inputs"].items():
                kept = set(source["pos_indices"] + source["neg_indices"])
                for row, sample, coherence in zip(source["rows"], source["samples"], source["coherence"]):
                    record = {"pole": pole, **row, "answer": sample["answer"], "trait_score": sample["score"],
                              "coherence": coherence, "kept": row["source_index"] in kept,
                              "source_sha256": source["source_sha256"]}
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        _write_json(vdir / "refilter_meta.json", {"task": "C18", "data_hashes": meta["data_hashes"],
                                                   "model": cfg["model"], "summary": summary})
        meta["artifact_hashes"] = {path.relative_to(output).as_posix(): sha256(path)
                                   for path in output.rglob("*") if path.is_file() and path.name != "run_meta.json"}
        meta["status"] = "completed"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error_type"] = type(exc).__name__
        raise
    finally:
        meta["end_time"] = now()
        _write_json(output / "run_meta.json", meta)
    print(f"重算完成：{output}")
    return output


def main():
    run_refilter(parse_args(add_arguments))


if __name__ == "__main__":
    main()
