"""C18：固定连贯方向剥离对照的向量计算与按题统计；不运行模型或评委。"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import itertools
import math
from statistics import stdev

import torch

from ..eval.calibration import assess_point
from ..eval.stats import benjamini_hochberg
from .data import BIG_FIVE


LAYER = 21
SEEDS = (1, 2, 3)
ARMS = ("zero", "original", "projected")
METRICS = ("trait", "coherence", "task")
COEFS = {"C": {"zero": 0., "original": -.225, "projected": -.225},
         "N": {"zero": 0., "original": .200, "projected": .200}}
UNIT_TOLERANCE = 1e-6
MIN_RETAINED_NORM = 1e-6
OUTPUT_TOLERANCE = 1e-6


def _finite_vector(value, name):
    try:
        vector = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} 必须为有限浮点向量") from error
    if (not vector.is_floating_point() or vector.ndim != 1 or vector.numel() < 2
            or not torch.isfinite(vector).all()):
        raise ValueError(f"{name} 必须为一维且至少含两个分量的有限浮点向量")
    return vector.double()


def project_vector(original_steer, q):
    """剥离后恢复原实际 L2 范数；输出 float32，保留原始单极注入符号。"""
    original, quality = _finite_vector(original_steer, "原实际注入"), _finite_vector(q, "q")
    if original.shape != quality.shape:
        raise ValueError("原注入与 q 形状不一致")
    original_norm, q_norm = float(original.norm()), float(quality.norm())
    if not math.isfinite(original_norm) or original_norm <= 0:
        raise ValueError("原实际注入范数必须为有限正数")
    if not math.isfinite(q_norm) or abs(q_norm - 1.) > UNIT_TOLERANCE:
        raise ValueError("q 必须已是单位方向；禁止自动修复非单位输入")
    # 只消除已通过单位检查的末位误差，不允许改变 q 的正向。
    q_unit = quality / q_norm
    unit_original = original / original_norm
    r = float(unit_original @ q_unit)
    remainder = unit_original - r * q_unit
    retained_norm = float(remainder.norm())
    if not math.isfinite(retained_norm) or retained_norm < MIN_RETAINED_NORM:
        raise ValueError("剥离后的单位向量剩余范数小于 1e-6，停止放大")
    projected = (remainder / retained_norm * original_norm).float()
    if not torch.isfinite(projected).all():
        raise ValueError("剥离向量转换为 float32 后非有限")
    actual_norm = float(projected.double().norm())
    if not math.isfinite(actual_norm) or actual_norm <= 0:
        raise ValueError("float32 剥离向量的实际范数无效")
    relative_norm_error = abs(actual_norm - original_norm) / original_norm
    projected_q_cosine = float((projected.double() / actual_norm) @ q_unit)
    if relative_norm_error > OUTPUT_TOLERANCE or abs(projected_q_cosine) > OUTPUT_TOLERANCE:
        raise ValueError("float32 剥离结果未满足等范数或近正交检查")
    return {"vector": projected,
            "audit": {"r": r, "retained_norm": retained_norm, "renorm_factor": 1. / retained_norm,
                      "original_actual_norm": original_norm, "projected_actual_norm": actual_norm,
                      "relative_norm_error": relative_norm_error, "projected_q_cosine": projected_q_cosine,
                      "q_input_norm": q_norm, "q_input_unit_tolerance": UNIT_TOLERANCE,
                      "min_retained_norm": MIN_RETAINED_NORM, "output_tolerance": OUTPUT_TOLERANCE,
                      "compute_dtype": "float64", "output_dtype": "float32",
                      "same_l2_injection": True, "same_trait_effect_established": False}}


def arm_phase(arm):
    if arm not in ARMS:
        raise ValueError("实验臂不在固定登记中")
    return "all" if arm == "zero" else "decode"


def validate_rows(rows, traits):
    """固定 1080 条身份：两维、双极、三臂、三种子、十题、三次采样。"""
    expected = {(d, p, a, seed, question, sample) for d in COEFS for p in ("high", "low")
                for a in ARMS for seed in SEEDS for question in range(10) for sample in range(3)}
    actual = []
    for row in rows:
        if not isinstance(row, dict) or any(type(row.get(key)) is not int for key in
                                           ("seed", "question_index", "sample_index", "layer")):
            raise ValueError("实验身份必须为整数，不能用布尔或浮点数")
        identity = tuple(row.get(key) for key in ("dim", "pole", "arm", "seed", "question_index", "sample_index"))
        if identity not in expected:
            raise ValueError("实验身份越过固定 calib 范围")
        actual.append(identity)
        target = BIG_FIVE[row["dim"]][row["pole"] == "low"]
        if (target not in traits or len(traits[target].get("questions", [])) not in (10, 20)
                or len(set(traits[target]["questions"][:10])) != 10
                or row.get("target") != target
                or row.get("q") != traits[target]["questions"][row["question_index"]]):
            raise ValueError("题目与十题极点 calib 来源不一致")
        if (row.get("split") != "calib" or row["layer"] != LAYER or row.get("axis") != "pole"
                or type(row.get("coef")) not in (int, float)
                or row["coef"] != COEFS[row["dim"]][row["arm"]]
                or row.get("steer_phase") != arm_phase(row["arm"])):
            raise ValueError("实验条件与固定剥离登记不一致")
    if len(actual) != 1080 or len(set(actual)) != 1080 or set(actual) != expected:
        raise ValueError("1080 条实验身份缺失或重复")


def _question_means_exact(rows, metric):
    """先聚合同题的三个采样及三个种子，精确保留边界分数。"""
    return {question: sum((Fraction(str(float(row[metric]))) for row in rows if row["question_index"] == question),
                          Fraction(0)) / sum(row["question_index"] == question for row in rows)
            for question in range(10)}


def _question_differences(left, right, metric):
    a, b = _question_means_exact(left, metric), _question_means_exact(right, metric)
    return [b[question] - a[question] for question in range(10)]


def exact_sign_flip(differences):
    """十题双侧精确符号翻转；复用既有检验定义，避免包反向依赖 experiments。"""
    if len(differences) != 10 or any(not math.isfinite(value) for value in differences):
        raise ValueError("精确符号翻转要求十个有限题目差值")
    values = [value if isinstance(value, Fraction) else Fraction(str(float(value))) for value in differences]
    observed = abs(sum(values, Fraction(0)))
    extreme = sum(abs(sum((value * sign for value, sign in zip(values, signs)), Fraction(0))) >= observed
                  for signs in itertools.product((-1, 1), repeat=10))
    return extreme / 1024


def _metrics(left, right):
    result = assess_point(left, right)["metrics"]
    if any(value is None for value in result.values()):
        raise ValueError("完整配对分数未得到三项统计")
    for metric in METRICS:
        # 保留 assess_point 的按题 bootstrap；绝对均值也用有理数保护 75 分边界。
        result[metric]["baseline"] = float(sum(_question_means_exact(left, metric).values(), Fraction(0)) / 10)
        result[metric]["injected"] = float(sum(_question_means_exact(right, metric).values(), Fraction(0)) / 10)
    return result


def _candidate(poles, target_pole):
    target = poles[target_pole]["trait"]
    checks = {"target_trait_gain_ge20": target["mean"] >= 20,
              "target_trait_ci_low_gt10": target["ci_low"] > 10}
    uncertainty = []
    for pole in ("high", "low"):
        checks[pole + "_coherence_absolute_ge75"] = poles[pole]["coherence"]["injected"] >= 75
        for metric in ("coherence", "task"):
            value = poles[pole][metric]
            checks[pole + "_" + metric + "_drop_le10"] = value["mean"] >= -10
            if value["ci_low"] <= -10 <= value["ci_high"]:
                uncertainty.append({"pole": pole, "metric": metric, "ci_low": value["ci_low"], "ci_high": value["ci_high"]})
    return {"checks": checks, "numeric_criteria_met": all(checks.values()),
            "failed_checks": [key for key, passed in checks.items() if not passed],
            "quality_boundary_uncertainty": uncertainty,
            "formal_e0_pass": False, "capability_evaluated": False}


def _method_checks(calib_candidate, target_quality):
    checks = {"projected_full_calib_criteria": calib_candidate,
              "target_coherence_improvement_ge5": target_quality["coherence"]["mean"] >= 5,
              "target_task_improvement_ge5": target_quality["task"]["mean"] >= 5}
    return {"checks": checks, "investment_criteria_met": all(checks.values()),
            "stop_current_dimension_projection": not all(checks.values()),
            "failed_checks": [key for key, passed in checks.items() if not passed]}


def analyze(rows, traits):
    """按十题做配对区间及检验；calib 候选、方法收益、正式验收分别报告。"""
    validate_rows(rows, traits)
    if any(type(row.get(metric)) not in (int, float) or not math.isfinite(row[metric]) or not 0 <= row[metric] <= 100
           for row in rows for metric in METRICS):
        raise ValueError("分析要求人格、连贯、任务三项评分全部有效，不丢弃缺分")
    if any(row.get("finish_reason") not in ("eos", "length") for row in rows):
        raise ValueError("完成原因必须保存为 eos 或 length")
    groups = {(dim, pole, arm): [row for row in rows if (row["dim"], row["pole"], row["arm"]) == (dim, pole, arm)]
              for dim in COEFS for pole in ("high", "low") for arm in ARMS}
    conditions, indexed = [], {}
    for dim in COEFS:
        for pole in ("high", "low"):
            for arm in ARMS[1:]:
                right = groups[dim, pole, arm]
                references = ("zero",) if arm == "original" else ("zero", "original")
                condition = {"dim": dim, "pole": pole, "arm": arm, "coef": COEFS[dim][arm],
                             "pooled": {}, "per_seed": [], "seed_stddev": {},
                             "finish_reasons": dict(Counter(row["finish_reason"] for row in right)),
                             "formal_e0_pass": False, "capability_evaluated": False}
                for reference in references:
                    condition["pooled"]["versus_" + reference] = _metrics(groups[dim, pole, reference], right)
                for seed in SEEDS:
                    seed_result = {"seed": seed, "formal_e0_pass": False}
                    for reference in references:
                        seed_result["versus_" + reference] = _metrics(
                            [row for row in groups[dim, pole, reference] if row["seed"] == seed],
                            [row for row in right if row["seed"] == seed])
                    condition["per_seed"].append(seed_result)
                for reference in references:
                    key = "versus_" + reference
                    condition["seed_stddev"][key] = {metric: stdev(item[key][metric]["mean"] for item in condition["per_seed"])
                                                      for metric in METRICS}
                indexed[dim, pole, arm] = condition
                conditions.append(condition)
    quality_tests = []
    for dim in COEFS:
        for pole in ("high", "low"):
            for metric in ("coherence", "task"):
                differences = _question_differences(groups[dim, pole, "original"], groups[dim, pole, "projected"], metric)
                quality_tests.append({"dim": dim, "pole": pole, "metric": metric,
                                      "question_differences": [float(value) for value in differences],
                                      "p": exact_sign_flip(differences),
                                      **indexed[dim, pole, "projected"]["pooled"]["versus_original"][metric]})
    correction = benjamini_hochberg([item["p"] for item in quality_tests], q=.05)
    for item, adjusted, rejected in zip(quality_tests, correction["adjusted_p"], correction["reject"]):
        item.update(adjusted_p=adjusted, reject=rejected,
                    quality_improvement_supported=item["ci_low"] > 0 and rejected, formal_e0_pass=False)
    candidates, candidate_index = [], {}
    for dim in COEFS:
        target_pole = "low" if dim == "C" else "high"
        for arm in ARMS[1:]:
            poles = {pole: indexed[dim, pole, arm]["pooled"]["versus_zero"] for pole in ("high", "low")}
            candidate = {"dim": dim, "arm": arm, "coef": COEFS[dim][arm], "target_pole": target_pole,
                         "pooled": _candidate(poles, target_pole), "per_seed": [], "formal_e0_pass": False,
                         "capability_evaluated": False}
            for i, seed in enumerate(SEEDS):
                seed_poles = {pole: indexed[dim, pole, arm]["per_seed"][i]["versus_zero"] for pole in ("high", "low")}
                candidate["per_seed"].append({"seed": seed, **_candidate(seed_poles, target_pole)})
            candidates.append(candidate)
            candidate_index[dim, arm] = candidate
    decisions = []
    for dim in COEFS:
        target_pole = "low" if dim == "C" else "high"
        projected = candidate_index[dim, "projected"]
        target = indexed[dim, target_pole, "projected"]
        versus_original = target["pooled"]["versus_original"]
        trait = versus_original["trait"]
        decisions.append({"dim": dim, "target_pole": target_pole,
                          "projected_calib_candidate": projected["pooled"]["numeric_criteria_met"],
                          **_method_checks(projected["pooled"]["numeric_criteria_met"], versus_original),
                          "target_quality_versus_original": {metric: versus_original[metric] for metric in ("coherence", "task")},
                          "target_trait_versus_original": {**trait,
                              "ci_within_minus5_plus5": trait["ci_low"] >= -5 and trait["ci_high"] <= 5,
                              "note": "相近效应区间只作预定描述；等注入范数不证明等人格效果。"},
                          "per_seed": [{"seed": seed,
                                        **_method_checks(projected["per_seed"][i]["numeric_criteria_met"],
                                                         target["per_seed"][i]["versus_original"])}
                                       for i, seed in enumerate(SEEDS)],
                          "formal_e0_pass": False, "capability_evaluated": False,
                          "note": "投入标准是后续研究筛查；统计改善须另看区间及八项 BH，不代表能力或正式 E0 通过。"})
    return {"conditions": conditions, "candidates": candidates, "method_decisions": decisions,
            "quality_comparisons": {"family": "2维×2极×连贯/任务，projected−original，共8项",
                                    "n_tests": 8, "q": .05,
                                    "method": "双侧精确符号翻转，枚举2^10种符号；统一BH校正",
                                    "assumption": "零假设下十道题差值独立、关于零对称，符号可交换；不保证一般相关结构下的BH控制。",
                                    "tests": quality_tests},
            "n_samples": len(rows), "seeds": list(SEEDS), "seed_stddev_ddof": 1,
            "n_questions": 10, "n_boot": 10000, "formal_e0_pass": False, "capability_evaluated": False,
            "notes": ["同题三采样和三种子先等权平均，区间及检验的独立单位为十题。",
                      "calib要求人格均值≥20、区间下限>10，两套题连贯绝对≥75且连贯和任务下降均≤10。",
                      "方法投入还要求目标题连贯和任务相对原方向均改善≥5；任一不满足即停止该维本轮剥离分支。",
                      "完整矩阵一次结束；不追加系数、层、评委重评或15/10分敏感性。",
                      "未运行正式test或有效decode能力检查；不能宣布E0或能力通过。"]}
