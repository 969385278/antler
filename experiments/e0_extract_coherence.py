"""C17 / E0 D-A：只用评委给已保存的提取回答补打连贯性分，不加载生成模型。

输入 vectors/<极点>_samples.json；输出 extract_coherence.json、
extract_coherence_samples.jsonl 和 run_meta.json，不改原始回答及人格评分。
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.judge import COHERENCE_PROMPT
from antler.traits.data import POLES

DEFAULT_POLES = ["dependable", "careless", "nervous", "calm"]
COHERENCE_THRESHOLD = 50


def add_arguments(ap):
    ap.add_argument("--poles", nargs="+", choices=POLES, default=DEFAULT_POLES,
                    help="需要补打分的极点，默认 dependable careless nervous calm")


def _score_is_valid(score):
    return (isinstance(score, (int, float)) and not isinstance(score, bool)
            and math.isfinite(score) and 0 <= score <= 100)


def load_inputs(out_path, poles):
    """先校验全部输入，避免后面的文件缺失或格式错误时已花费评委费用。"""
    if not poles or len(set(poles)) != len(poles) or any(p not in POLES for p in poles):
        raise ValueError("极点必须是大五十个极点中非空且不重复的列表")
    inputs = {}
    for pole in poles:
        path = (Path(out_path) / "vectors" / f"{pole}_samples.json").resolve()
        raw = path.read_bytes()
        rows = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError(f"{path}: 原始提取回答必须是列表")
        for i, row in enumerate(rows):
            if (not isinstance(row, dict) or row.get("side") not in ("pos", "neg")
                    or not isinstance(row.get("question"), str)
                    or not isinstance(row.get("answer"), str) or "score" not in row
                    or (row["score"] is not None and not _score_is_valid(row["score"]))):
                raise ValueError(f"{path}: 第 {i} 条样本格式无效，需要 side/question/answer/score")
        inputs[pole] = {"source_file": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "samples": rows}
    return inputs


def summarize(scores):
    """均值及低分比例只以有效评分为分母；缺失值独立统计。"""
    valid = [s for s in scores if s is not None]
    below = sum(s < COHERENCE_THRESHOLD for s in valid)
    return {"n_total": len(scores), "n_valid": len(valid), "n_missing": len(scores) - len(valid),
            "n_below_threshold": below,
            "mean_coherence": sum(valid) / len(valid) if valid else None,
            "below_threshold_fraction": below / len(valid) if valid else None}


def score_inputs(inputs, judge):
    """保留源文件顺序和下标，便于 C18 还原系统提示并成对过滤。"""
    summary = {"schema_version": 1, "threshold": COHERENCE_THRESHOLD,
               "below_threshold_denominator": "n_valid",
               "missing_score_policy": "缺失分不参与均值和低分比例；无有效分时两者为 null",
               "poles": {}}
    records = []
    for pole, source in inputs.items():
        rows = source["samples"]
        scores = judge.coherence_scores([r["question"] for r in rows], [r["answer"] for r in rows]) if rows else []
        if len(scores) != len(rows):
            raise ValueError(f"{pole}: 连贯性评分数量与原始样本数不一致")
        if any(s is not None and not _score_is_valid(s) for s in scores):
            raise ValueError(f"{pole}: 连贯性分数必须是 0–100 的有限数值或 None")
        provenance = {k: source[k] for k in ("source_file", "source_sha256")}
        groups = {side: summarize([score for row, score in zip(rows, scores) if row["side"] == side])
                  for side in ("pos", "neg")}
        summary["poles"][pole] = {**provenance, **groups}
        for i, (row, score) in enumerate(zip(rows, scores)):
            records.append({"pole": pole, "source_index": i, **provenance, "side": row["side"],
                            "question": row["question"], "answer": row["answer"],
                            "trait_score": row["score"], "coherence": score})
        for side, group in groups.items():
            print(f"[{pole}/{side}] 有效 {group['n_valid']}/{group['n_total']}，"
                  f"缺失 {group['n_missing']}，平均连贯性 {group['mean_coherence']}，"
                  f"低于 50 比例 {group['below_threshold_fraction']}")
    return summary, records


def _redacted(value):
    """不把配置中的认证字段或 URL 内嵌凭据写入元数据。"""
    private = {"api_key", "judge_api_key", "hf_token", "token", "access_token", "password",
               "secret", "authorization", "proxy_authorization", "cookie", "set_cookie"}
    if isinstance(value, dict):
        return {k: ("[已隐藏]" if str(k).lower().replace("-", "_") in private else _redacted(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(v) for v in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        parts = urlsplit(value)
        # 记录端点身份即可，不保留用户名、密码、查询参数或片段。
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        f.write("\n")


@contextmanager
def record_run(args, cfg, judge, inputs):
    """每次补打分记录输入哈希、配置和完成状态，不写入异常中的潜在凭据。"""
    def git(*argv):
        try:
            return subprocess.check_output(["git", *argv], cwd=ROOT, text=True, encoding="utf-8",
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    now = lambda: datetime.now(timezone.utc).isoformat()
    state = git("status", "--porcelain")
    config_path = Path(args.config)
    meta = {"task": "C17", "experiment": "e0_extract_coherence", "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(state) if state is not None else None,
            "config_path": str(config_path), "config": _redacted(cfg), "argv": _redacted(sys.argv[1:]),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None,
            "start_time": now(), "end_time": None, "status": "running",
            "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "judge": _redacted({k: getattr(judge, k, None) for k in ("model", "base_url", "max_tokens", "extra_body")}),
            "seed": cfg.get("seed", 0), "generation_performed": False, "model_loaded": False,
            "poles": list(inputs), "threshold": COHERENCE_THRESHOLD,
            "data_hashes": {source["source_file"]: source["source_sha256"] for source in inputs.values()},
            "coherence_prompt_sha256": hashlib.sha256(COHERENCE_PROMPT.encode("utf-8")).hexdigest()}
    path = cfg["out_path"] / "run_meta.json"
    _write_json(path, meta)
    try:
        yield
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error_type"] = type(exc).__name__
        raise
    else:
        meta["status"] = "completed"
    finally:
        meta["end_time"] = now()
        _write_json(path, meta)


def main():
    args = parse_args(add_arguments)
    cfg, _, _, judge = setup(args, need_model=False)
    inputs = load_inputs(cfg["out_path"], args.poles)
    if judge is None:
        raise ValueError("补打连贯性分需要启用评委，请检查 use_judge 和评委环境变量")
    summary_path = cfg["out_path"] / "extract_coherence.json"
    samples_path = cfg["out_path"] / "extract_coherence_samples.jsonl"
    with record_run(args, cfg, judge, inputs):
        summary, records = score_inputs(inputs, judge)
        # 全部评分成功后才覆盖结果文件；API 异常时保留已有结果和失败元数据。
        _write_json(summary_path, summary)
        with open(samples_path, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"汇总：{summary_path}")
    print(f"逐条连贯性分：{samples_path}")


if __name__ == "__main__":
    main()
