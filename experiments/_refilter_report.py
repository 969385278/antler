"""C18/C20：汇总诊断结果、导出任务理解盲标表，并打包可分享的实验产物。"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import zipfile
from pathlib import Path


VARIANTS = ("original", "paired", "cohfilt")
MEASUREMENTS = {
    "c_high": ("C", "high", (0.0, 0.2)),
    "c_low": ("C", "low", (-0.2, 0.0)),
    "n_high": ("N", "high", (0.0, 0.2, 0.25)),
    "n_low": ("N", "low", (-0.2, 0.0)),
}
VARIANT_NAMES = {"original": "原始向量", "paired": "仅配对过滤", "cohfilt": "配对＋连贯性过滤"}
SCORE_NAMES = ("trait", "coherence", "task")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _mean(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _number(value, *, signed=False):
    if value is None:
        return "缺失"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _conditions(run_dir):
    """按约定条件读逐条回答；均值从有效原始分计算，不把缺失当作零。"""
    conditions = []
    for variant in VARIANTS:
        for measurement, (dim, pole, coefs) in MEASUREMENTS.items():
            path = run_dir / "steps" / f"{variant}_{measurement}" / "calibration_judge_samples.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                       if line.strip()]
            by_coef = {}
            for record in records:
                coef = record["coef"]
                if (record["layer"] != 16 or record["dim"] != dim or record["pole"] != pole
                        or coef not in coefs or coef in by_coef):
                    raise ValueError(f"条件不匹配或重复：{path}")
                samples = record["result"]["samples"]
                if not samples:
                    raise ValueError(f"条件缺少回答：{path} coef={coef}")
                for sample in samples:
                    if not isinstance(sample.get("q"), str) or not isinstance(sample.get("a"), str):
                        raise ValueError(f"缺少有效问题或回答：{path}")
                    if sample.get("finish_reason") not in ("eos", "length"):
                        raise ValueError(f"缺少有效停止原因：{path}")
                    for name in SCORE_NAMES:
                        if name not in sample:
                            raise ValueError(f"缺少逐条 {name} 字段：{path}")
                        value = sample[name]
                        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                                  or not math.isfinite(value) or not 0 <= value <= 100):
                            raise ValueError(f"无效 {name} 分数：{path}")
                means = {name: _mean([s[name] for s in samples]) for name in SCORE_NAMES}
                counts = {name: sum(s[name] is not None for s in samples) for name in SCORE_NAMES}
                by_coef[coef] = {
                    "variant": variant, "measurement": measurement, "layer": 16, "dim": dim,
                    "pole": pole, "coef": coef, "samples": samples, "means": means, "counts": counts,
                    "n_length": sum(s["finish_reason"] == "length" for s in samples),
                    "source": path.relative_to(run_dir).as_posix(),
                }
            if set(by_coef) != set(coefs):
                raise ValueError(f"缺少完整系数条件：{path}")
            baseline = by_coef[0.0]["means"]
            for coef in coefs:
                condition = by_coef[coef]
                condition["deltas"] = {
                    name: None if condition["means"][name] is None or baseline[name] is None
                    else condition["means"][name] - baseline[name] for name in SCORE_NAMES
                }
                conditions.append(condition)
    return conditions


def _blind_table(run_dir, conditions):
    """每种向量的九个条件各抽一条，再各补一条；不在表内暴露条件或评委分。"""
    rng = random.Random(0)
    selected = []
    used_qa = set()
    for variant in VARIANTS:
        group = [condition for condition in conditions if condition["variant"] == variant]
        used = set()
        for ci, condition in enumerate(group):
            unseen = [si for si, sample in enumerate(condition["samples"])
                      if (sample["q"], sample["a"]) not in used_qa]
            si = rng.choice(unseen or list(range(len(condition["samples"]))))
            selected.append((condition, si))
            used.add((ci, si))
            sample = condition["samples"][si]
            used_qa.add((sample["q"], sample["a"]))
        available = [(ci, si) for ci, condition in enumerate(group)
                     for si in range(len(condition["samples"])) if (ci, si) not in used]
        if not available:
            raise ValueError(f"{variant} 样本不足，无法额外抽取第十条盲标回答")
        unseen = [(ci, si) for ci, si in available
                  if (group[ci]["samples"][si]["q"], group[ci]["samples"][si]["a"]) not in used_qa]
        ci, si = rng.choice(unseen or available)
        selected.append((group[ci], si))
        sample = group[ci]["samples"][si]
        used_qa.add((sample["q"], sample["a"]))
    rng.shuffle(selected)
    mapping = {}
    rows = []
    for i, (condition, si) in enumerate(selected, 1):
        identity = f"T{i:03d}"
        sample = condition["samples"][si]
        rows.append({"id": identity, "question": sample["q"], "answer": sample["a"], "task_human": ""})
        mapping[identity] = {
            "condition": {k: condition[k] for k in ("variant", "measurement", "layer", "dim", "pole", "coef")},
            "source": condition["source"], "sample_index": si, "task_judge": sample["task"],
        }
    key = {"schema_version": 1, "seed": 0, "n_samples": 30, "task_judge_validated": False,
           "n_unique_qa": len(used_qa), "n_duplicate_qa": 30 - len(used_qa),
           "instructions": "作者先独立完成 task_blind.csv 的 30 条任务理解评分，再查看本映射。"
                           "正式使用前须按 RESEARCH_SPEC 4.6 计算与评委的 Spearman 相关，要求 rho >= 0.6；"
                           "缺失评分必须报告，不能按零补齐。当前任务分仅用于探索性诊断。",
           "samples": mapping}
    csv_path, key_path = run_dir / "task_blind.csv", run_dir / "task_blind_key.json"
    if _same_blind_identity(csv_path, key_path, rows, key):
        # 作者可能已填入 task_human；身份未变时连文件字节也保持不变。
        return
    existing = [p for p in (csv_path, key_path) if p.exists()]
    if existing:
        index = 1
        while (run_dir / "previous_blind" / f"{index:04d}").exists():
            index += 1
        backup = run_dir / "previous_blind" / f"{index:04d}"
        backup.mkdir(parents=True)
        for path in existing:
            shutil.copyfile(path, backup / path.name)
        print(f"[盲标] 抽样身份变化或旧表不完整，已备份旧标注：{backup}")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "task_human"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _same_blind_identity(csv_path, key_path, expected_rows, expected_key):
    """允许人工改分、调整行序及附加备注列；问题、回答、ID 和条件映射必须吻合。"""
    if not csv_path.is_file() or not key_path.is_file():
        return False
    try:
        key = _read_json(key_path)
        if not isinstance(key, dict) or key.get("samples") != expected_key["samples"]:
            return False
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"id", "question", "answer", "task_human"}.issubset(reader.fieldnames or []):
                return False
            rows = list(reader)
        expected = {r["id"]: (r["question"], r["answer"]) for r in expected_rows}
        actual = {r["id"]: (r["question"], r["answer"]) for r in rows}
        return len(rows) == len(actual) == len(expected_rows) and actual == expected
    except (ValueError, TypeError, KeyError, csv.Error):
        return False


def _blind_instructions(run_dir):
    """盲标说明独立于含实验条件的汇总和映射文件。"""
    text = """# 任务理解盲标说明

先只阅读本说明和 task_blind.csv。独立完成全部 30 条评分后，再打开 summary.md 或 task_blind_key.json；后两份文件包含实验条件和评委分数。

在 task_human 列填写 0–100 的数字。只判断回答是否理解并回应了题目实际描述的情境。不要评价建议是否好、明智、友善、负责，也不要评价回答的人格或语气。马虎、懒惰、焦虑、自私的回答也可能完全理解题目。

- 0 分：误读了情境、回答了另一个问题，或编造与题目矛盾的事实。
- 50 分：部分回应了情境，但遗漏或歪曲关键细节，例如忽略明确的期限或限制。
- 100 分：清楚理解情境，回应了实际所问的问题，无论其人格如何。

中间情况可在相应范围内打分。保持 id、question、answer 不变；完成后保存 CSV。相同问题与回答已尽量避免重复，样本不足时仍可能出现重复；请按同一标准独立评分。

本轮任务理解分尚未验证。按 RESEARCH_SPEC 第 4.6 节，正式使用前需完成 30 条作者盲标，并检查与评委的 Spearman ρ ≥ 0.6；当前导出不代表已经通过。一致性应在评分完成后计算，缺失评分不能按零补齐。
"""
    (run_dir / "task_blind_instructions.md").write_text(text, encoding="utf-8", newline="\n")


def _summary(run_dir, conditions):
    lines = ["# C18 / C20 诊断汇总", "",
             "本轮为 calib 题上的探索性诊断，不构成独立 test 验收。任务理解评委尚未经作者盲标验证。",
             "每题生成 3 条回答，不是三个独立种子实验。同系数不等于相同人格强度；请结合原始回答比较。",
             "均值仅使用对应量表的有效分；人格、连贯性、任务理解分别计数，缺失分不按零处理。",
             "Δ 相对于同一种向量、同维度、同极点的零注入基线；length 比例以全部回答为分母。", "",
             "## 人格、连贯性与任务理解", "",
             "| 向量 | 维度/极点 | 系数 | 人格 | Δ人格 | 连贯性 | Δ连贯性 | 任务理解 | Δ任务理解 | 有效数 人格/连贯/任务 | length |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for c in conditions:
        cells = [VARIANT_NAMES[c["variant"]], f"{c['dim']}/{c['pole']}", f"{c['coef']:+g}"]
        for name in SCORE_NAMES:
            cells += [_number(c["means"][name]), _number(c["deltas"][name], signed=True)]
        n = len(c["samples"])
        cells += ["/".join(str(c["counts"][name]) for name in SCORE_NAMES),
                  f"{c['n_length']}/{n} ({100 * c['n_length'] / n:.1f}%)"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## 第 16 层 C–N 方向余弦相似度", "",
              "余弦相似度用于检查两个方向是否重叠，变化本身不能证明人格控制改善。", "",
              "| 重算方式 | 原始 C–N | 重算后 C–N |", "|---|---:|---:|"]
    summaries = {}
    for variant in ("paired", "cohfilt"):
        summary = _read_json(run_dir / "steps" / f"refilter_{variant}" / "refilter_summary.json")
        summaries[variant] = summary
        lines.append(f"| {VARIANT_NAMES[variant]} | {_number(summary['cn_cosine_before'][16])} | "
                     f"{_number(summary['cn_cosine_after'][16])} |")
    lines += ["", "## 提取样本保留数量", "",
              "仅配对过滤是匹配对照；配对＋连贯性过滤在相同人格阈值和配对规则上增加质量筛选。", "",
              "| 重算方式 | 极点 | 原始回答 | 人格筛选合格对数 | 最终保留对数 | 最终保留回答数 |",
              "|---|---|---:|---:|---:|---:|"]
    for variant, summary in summaries.items():
        for pole, counts in sorted(summary["poles"].items()):
            kept = counts["n_kept_pairs"]
            lines.append(f"| {VARIANT_NAMES[variant]} | {pole} | {counts['n_total']} | "
                         f"{counts['n_trait_pairs']} | {kept} | {2 * kept} |")
    lines += ["", "## 任务理解人工盲标", "",
              "先阅读独立的 task_blind_instructions.md，再打开 task_blind.csv，只填写 task_human（0–100）；"
              "盲标者完成前不应查看本汇总及 task_blind_key.json。",
              "重跑时，抽样身份相同会保留已有人工评分；身份变化时旧表保存在 previous_blind/，不会进入结果包。",
              "30 条按三种向量的九个条件分层抽取，再每种向量补一条；固定种子 0 并打乱顺序。"
              "优先避免相同问题＋回答重复；唯一数量及重复数量记录在 key 中。",
              "按照 RESEARCH_SPEC 第 4.6 节，正式使用任务理解分前，作者应完成 30 条盲标，"
              "与评委的 Spearman ρ ≥ 0.6；不一致时先检查评分提示。当前未计算人工一致性，也未宣称通过。",
              "人格变化的大小、额外理解错误与截断情况应一起检查，不能仅凭连贯性均分判断方向好坏。", ""]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _archive(run_dir):
    """只打包报告、结果和日志；不带模型、配置、缓存、凭据或历史备份。"""
    allowed_names = {"calibration_judge.csv", "calibration_judge_samples.jsonl", "run_meta.json",
                     "refilter_summary.json", "refilter_samples.jsonl", "refilter_meta.json",
                     "extract_coherence.json", "extract_coherence_samples.jsonl"}
    paths = [run_dir / name for name in ("summary.md", "run_meta.json", "task_blind.csv", "task_blind_key.json",
                                        "task_blind_instructions.md")]
    for path in (run_dir / "steps").rglob("*"):
        if path.is_file() and path.name in allowed_names:
            paths.append(path)
    paths.extend((run_dir / "attempts").glob("*/log.txt"))
    archive = run_dir / "results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(set(paths)):
            relative = path.relative_to(run_dir)
            if any(any(word in part.lower() for word in ("cache", "config", "secret", "previous"))
                   for part in relative.parts):
                continue
            if not path.resolve().is_relative_to(run_dir.resolve()) or path.is_symlink():
                continue
            output.write(path, relative.as_posix())
    return archive


def finish_report(run_dir: Path) -> Path:
    """所有步骤完成后生成汇总、30 条盲标表、可分享 ZIP；返回 ZIP 路径。"""
    run_dir = Path(run_dir)
    conditions = _conditions(run_dir)
    _summary(run_dir, conditions)
    _blind_table(run_dir, conditions)
    _blind_instructions(run_dir)
    return _archive(run_dir)
