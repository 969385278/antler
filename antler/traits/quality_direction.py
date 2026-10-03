"""C18：四极点旧回答上的组内连贯性相关方向；纯 CPU、无模型调用。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import numpy as np

from .vectors import select_paired_samples


POLES = ("dependable", "careless", "nervous", "calm")
SIDES = ("pos", "neg")
FOLDS = ("A", "B")
TARGETS = ("careless", "nervous")
DESIGN_COLUMNS = ([f"intercept:{p}:{s}" for p in POLES for s in SIDES]
                  + [f"trait_difference:{p}" for p in POLES] + ["log1p_token_difference"])


def _array(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().double().numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限数值")
    return array


def _unit(vector, name):
    vector = _array(vector, name)
    if vector.ndim != 1 or not vector.size:
        raise ValueError(f"{name} 必须是一维向量")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} 范数为零或无效")
    return vector / norm


def _cosine(left, right):
    left, right = _unit(left, "余弦左向量"), _unit(right, "余弦右向量")
    if left.shape != right.shape:
        raise ValueError("余弦向量形状不一致")
    return float(np.clip(left @ right, -1.0, 1.0))


def quality_question_folds(rows: list[dict]) -> list[dict]:
    """只读身份生成固定分题结果；须由入口在模型前保存及哈希。"""
    if not isinstance(rows, list) or len(rows) != 1600:
        raise ValueError("本轮固定四极点各 400 条，共 1600 条")
    by_pole = defaultdict(list)
    groups = defaultdict(set)
    all_questions = set()
    for row in rows:
        p, q, side = row.get("pole"), row.get("question"), row.get("side")
        if p not in POLES or side not in SIDES or not isinstance(q, str) or not q:
            raise ValueError("回答的 pole、side 或 question 身份无效")
        for key, limit in (("source_index", 400), ("pair_index", 5), ("rollout", 2)):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
                raise ValueError(f"{key} 身份无效")
        condition = (p, q, row["pair_index"], side)
        if row["rollout"] in groups[condition]:
            raise ValueError("同一条件出现重复 rollout 身份")
        groups[condition].add(row["rollout"])
        by_pole[p].append(row)
    assignment = {}
    for p in POLES:
        part = by_pole[p]
        questions = {r["question"] for r in part}
        if len(part) != 400 or {r["source_index"] for r in part} != set(range(400)):
            raise ValueError(f"{p} 的 source_index 必须恰好覆盖 0..399")
        if len(questions) != 20 or all_questions & questions:
            raise ValueError("每极点须为 20 道题，四极点问题不得跨组重复")
        all_questions.update(questions)
        for q in questions:
            for pair in range(5):
                for side in SIDES:
                    if groups[(p, q, pair, side)] != {0, 1}:
                        raise ValueError("题目未包含五提示、两侧、两次生成的完整身份")
        hashes = {q: hashlib.sha256(("quality-direction-v1\n" + p + "\n" + q).encode("utf-8")).hexdigest()
                  for q in questions}
        for position, q in enumerate(sorted(questions, key=lambda question: hashes[question])):
            assignment[(p, q)] = (FOLDS[position % 2], hashes[q])
    result = []
    for row in rows:
        fold, digest = assignment[(row["pole"], row["question"])]
        if ("fold" in row and row["fold"] != fold
                or "question_sha256" in row and row["question_sha256"] != digest):
            raise ValueError("已有折分与固定问题哈希不一致")
        result.append({**row, "fold": fold, "question_sha256": digest})
    return result


def _validated_activations(rows, activations):
    acts = _array(activations, "回答平均激活")
    if acts.ndim != 2 or acts.shape[0] != len(rows) or acts.shape[1] < 1:
        raise ValueError("回答激活必须为 (1600, d)")
    return acts


def prepare_quality_pairs(rows, activations, response_tokens, s_layer):
    """固定 rollout1−rollout0，构造 800 对及 13 列协变量，不筛分差。"""
    rows = quality_question_folds(rows)
    acts = _validated_activations(rows, activations)
    tokens = _array(response_tokens, "回答 token 数")
    if tokens.shape != (len(rows),) or np.any(tokens < 1) or np.any(tokens != np.floor(tokens)):
        raise ValueError("回答 token 数必须为对应每条回答的正整数")
    if isinstance(s_layer, bool) or not isinstance(s_layer, (float, int, np.number)) or not np.isfinite(s_layer) or s_layer <= 0:
        raise ValueError("层尺度必须为正有限数值")
    scores, coherence = [], []
    identities = defaultdict(dict)
    for index, row in enumerate(rows):
        for key, target in (("score", scores), ("coherence", coherence)):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"{key} 必须为 0..100 有限分数；不允许删除缺失样本")
            target.append(float(value))
        identities[(row["pole"], row["question"], row["pair_index"], row["side"])][row["rollout"]] = index
    X, y, A, pairs = [], [], [], []
    for key in sorted(identities, key=lambda k: (POLES.index(k[0]), k[1], k[2], SIDES.index(k[3]))):
        p, question, pair, side = key
        i0, i1 = identities[key][0], identities[key][1]
        design = np.zeros(13, dtype=np.float64)
        design[2 * POLES.index(p) + SIDES.index(side)] = 1.0
        design[8 + POLES.index(p)] = scores[i1] - scores[i0]
        design[12] = np.log1p(tokens[i1]) - np.log1p(tokens[i0])
        X.append(design)
        y.append(coherence[i1] - coherence[i0])
        A.append((acts[i1] - acts[i0]) / float(s_layer))
        pairs.append({"pole": p, "question": question, "pair_index": pair, "side": side,
                      "fold": rows[i0]["fold"], "question_sha256": rows[i0]["question_sha256"],
                      "row0": i0, "row1": i1, "source_index0": rows[i0]["source_index"],
                      "source_index1": rows[i1]["source_index"]})
    arrays = {"X": np.asarray(X), "y": np.asarray(y), "A": np.asarray(A)}
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("作差或尺度换算后包含非有限数值")
    return {**arrays, "pair_rows": pairs}


def fit_quality_direction(X, y, A, *, weights=None):
    """FWL 残差回归；方向不翻转，所有 nuisance 和标量仅由训练行估计。"""
    X, y, A = _array(X, "X"), _array(y, "y"), _array(A, "A")
    if X.ndim != 2 or X.shape[1] != 13 or y.shape != (len(X),) or A.ndim != 2 or len(A) != len(X):
        raise ValueError("拟合要求 X=(n,13)、y=(n,)、A=(n,d)")
    w = np.ones(len(X)) if weights is None else _array(weights, "训练权重")
    if w.shape != (len(X),) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("训练权重无效")
    root_w = np.sqrt(w)
    Xw = X * root_w[:, None]
    rank = int(np.linalg.matrix_rank(Xw))
    if rank != 13:
        raise ValueError(f"设计矩阵秩不足：{rank}/13")
    # 只在低维 nuisance 空间求解；不反演 d 维激活协方差。
    beta, _, _, singular = np.linalg.lstsq(Xw, np.column_stack((y, A)) * root_w[:, None], rcond=None)
    beta_y, beta_A = beta[:, 0], beta[:, 1:]
    ry, RA = y - X @ beta_y, A - X @ beta_A
    denominator = float((w * ry) @ ry)
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps * max(1.0, float((w * y) @ y)):
        raise ValueError("连贯性残差为零或数值无效")
    b = RA.T @ (w * ry) / denominator
    q = _unit(b, "连贯方向")
    projection = RA @ q
    projection_energy = float((w * projection) @ projection)
    if not np.isfinite(projection_energy) or projection_energy <= 1e-12:
        raise ValueError("方向投影残差为零或数值无效")
    slope = float((w * projection) @ ry / projection_energy)
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError("训练标量斜率不是有限正数；不得翻转方向")
    return {"q": q, "b": b, "beta_y": beta_y, "beta_A": beta_A,
            "calibration_slope": slope, "rank": rank,
            "condition_number": float(singular[0] / singular[-1]),
            "residual_y_energy": denominator, "residual_y": ry, "residual_A": RA}


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(left, right):
    """精确平均秩；常量组记 None，不能伪造成零相关。"""
    left, right = _array(left, "相关左列"), _array(right, "相关右列")
    if left.ndim != 1 or left.shape != right.shape or len(left) < 2:
        return None
    left, right = _ranks(left), _ranks(right)
    left, right = left - left.mean(), right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator <= 0 else float(np.clip(left @ right / denominator, -1, 1))


def crossfit_quality_direction(X, y, A, pair_rows, *, weights=None):
    """A 训练/B 评估及 B 训练/A 评估；评估行不参与 nuisance 或缩放。"""
    X, y, A = _array(X, "X"), _array(y, "y"), _array(A, "A")
    if len(pair_rows) != len(X):
        raise ValueError("配对身份数量与矩阵不一致")
    fold = np.asarray([r["fold"] for r in pair_rows])
    if set(fold) != set(FOLDS):
        raise ValueError("必须包含固定的 A/B 两折")
    w = np.ones(len(X)) if weights is None else _array(weights, "跨折权重")
    if w.shape != (len(X),) or np.any(w < 0) or np.any(w != np.floor(w)):
        raise ValueError("跨折权重必须为非负整数整题重复次数")
    observed, predicted = np.empty(len(y)), np.empty(len(y))
    fits, statistics = {}, {}
    for train in FOLDS:
        training, evaluation = fold == train, fold != train
        fitted = fit_quality_direction(X[training], y[training], A[training], weights=w[training])
        ry = y[evaluation] - X[evaluation] @ fitted["beta_y"]
        # 两折共用“预测连贯分差”单位，不将原始投影直接拼接。
        prediction = (A[evaluation] @ fitted["q"] - X[evaluation] @ (fitted["beta_A"] @ fitted["q"])) * fitted["calibration_slope"]
        if not np.isfinite(prediction).all() or not np.isfinite(ry).all():
            raise ValueError("跨折预测出现非有限数值")
        observed[evaluation], predicted[evaluation] = ry, prediction
        repeated = np.repeat(np.arange(evaluation.sum()), w[evaluation].astype(int))
        statistics[f"train_{train}_evaluate_{'B' if train == 'A' else 'A'}"] = {
            "rho": spearman_correlation(ry[repeated], prediction[repeated]),
            "n_pairs": len(repeated), "rank": fitted["rank"],
            "condition_number": fitted["condition_number"],
            "calibration_slope": fitted["calibration_slope"]}
        fits[train] = fitted
    repeated = np.repeat(np.arange(len(y)), w.astype(int))
    rho = spearman_correlation(observed[repeated], predicted[repeated])
    if rho is None or any(s["rho"] is None for s in statistics.values()):
        raise ValueError("合并或单折未见题残差相关无定义")
    by_group = {}
    for p in POLES:
        for side in SIDES:
            group = np.asarray([r["pole"] == p and r["side"] == side for r in pair_rows])
            indices = np.repeat(np.flatnonzero(group), w[group].astype(int))
            by_group[f"{p}/{side}"] = {"n_pairs": len(indices), "rho": spearman_correlation(observed[indices], predicted[indices])}
    return {"fits": fits, "oof_y": observed, "oof_prediction": predicted,
            "rho": rho, "folds": statistics, "pole_side": by_group}


def bootstrap_question_weights(pair_rows, rng):
    """分 fold/pole 重抽整题；同题十对始终共享一个重复次数。"""
    counts = {}
    for fold in FOLDS:
        for pole in POLES:
            questions = sorted({r["question"] for r in pair_rows if r["fold"] == fold and r["pole"] == pole})
            if not questions:
                raise ValueError("bootstrap 分层没有题目")
            draw = rng.integers(0, len(questions), size=len(questions))
            multiplicity = np.bincount(draw, minlength=len(questions))
            for question, count in zip(questions, multiplicity):
                counts[(fold, pole, question)] = int(count)
    return np.asarray([counts[(r["fold"], r["pole"], r["question"])] for r in pair_rows], dtype=np.int64)


def bootstrap_quality_direction(X, y, A, pair_rows, *, bootstrap_samples=1000, seed=0):
    """每次整题重抽后重新拟合两折；失败留档，存在失败时不生成通过区间。"""
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples 必须为正整数")
    rng, estimates, failures = np.random.default_rng(seed), [], []
    for repeat in range(bootstrap_samples):
        weights = bootstrap_question_weights(pair_rows, rng)
        try:
            estimates.append(crossfit_quality_direction(X, y, A, pair_rows, weights=weights)["rho"])
        except (ValueError, np.linalg.LinAlgError) as error:
            estimates.append(None)
            failures.append({"replicate": repeat, "error": str(error)})
            break
    interval = None if failures else [float(v) for v in np.quantile(estimates, [0.025, 0.975])]
    return {"samples": bootstrap_samples, "seed": seed, "unit": "fold/pole/question (10 pairs)",
            "refit_each_replicate": True, "rho_values": estimates, "failures": failures,
            "attempted": len(estimates), "rho_ci95": interval, "complete": not failures}


def reconstruct_paired_vectors(rows, activations):
    """同批全部激活按原 paired 规则重建 full/A/B，保留单极 pos−neg 符号。"""
    rows = quality_question_folds(rows)
    acts = _validated_activations(rows, activations)
    result = {}
    for pole in POLES:
        ordered = sorted((i for i, row in enumerate(rows) if row["pole"] == pole), key=lambda i: rows[i]["source_index"])
        result[pole] = {}
        for subset in ("full", "A", "B"):
            indices = [i for i in ordered if subset == "full" or rows[i]["fold"] == subset]
            samples = [rows[i] for i in indices]
            pos, neg = select_paired_samples(samples, min_coherence=None)
            if len(pos) < 5:
                raise ValueError(f"{pole}/{subset}: paired 过滤后不足 5 对")
            vector = acts[[indices[i] for i in pos]].mean(axis=0) - acts[[indices[i] for i in neg]].mean(axis=0)
            _unit(vector, f"{pole}/{subset}")
            result[pole][subset] = {"vector": vector, "n_pairs": len(pos),
                                    "source_pairs": [[samples[a]["source_index"], samples[b]["source_index"]] for a, b in zip(pos, neg)]}
    return result


def analyze_quality_direction(rows, activations, response_tokens, s_layer, reference_vectors,
                              *, bootstrap_samples=1000, seed=0):
    """有限登记的整套统计；失败抛错由入口留档，成功包含通过和未通过结果。"""
    pairs = prepare_quality_pairs(rows, activations, response_tokens, s_layer)
    vectors = reconstruct_paired_vectors(rows, activations)
    replay_checks = {}
    for pole in POLES:
        if pole not in reference_vectors:
            raise ValueError(f"缺少原 paired 向量：{pole}")
        reference = _array(reference_vectors[pole], f"原 {pole} 向量")
        replay = vectors[pole]["full"]["vector"]
        cosine = _cosine(replay, reference)
        relative_norm_error = float(abs(np.linalg.norm(replay) - np.linalg.norm(reference)) / np.linalg.norm(reference))
        relative_vector_error = float(np.linalg.norm(replay - reference) / np.linalg.norm(reference))
        replay_checks[pole] = {"cosine": cosine, "relative_norm_error": relative_norm_error,
                               "relative_vector_error": relative_vector_error,
                               "n_pairs": {s: vectors[pole][s]["n_pairs"] for s in ("full", "A", "B")},
                               "passed": cosine >= .995 and relative_norm_error <= .05}
    if not all(item["passed"] for item in replay_checks.values()):
        raise ValueError(f"重放与既有 paired 向量不一致，停止统计解释：{replay_checks}")
    X, y, A = pairs["X"], pairs["y"], pairs["A"]
    full = fit_quality_direction(X, y, A)
    cross = crossfit_quality_direction(X, y, A, pairs["pair_rows"])
    bootstrap = bootstrap_quality_direction(X, y, A, pairs["pair_rows"], bootstrap_samples=bootstrap_samples, seed=seed)
    stability = _cosine(cross["fits"]["A"]["q"], cross["fits"]["B"]["q"])
    supported = (bootstrap["complete"] and bootstrap["rho_ci95"][0] > 0
                 and all(item["rho"] > 0 for item in cross["folds"].values()) and stability >= .30)
    targets = {}
    for pole in TARGETS:
        # 正式阈值对实际保存、用于注入的向量判定；重建容差不能改变边界。
        full_cosine = _cosine(full["q"], reference_vectors[pole])
        replayed_full_cosine = _cosine(full["q"], vectors[pole]["full"]["vector"])
        qa_vb = _cosine(cross["fits"]["A"]["q"], vectors[pole]["B"]["vector"])
        qb_va = _cosine(cross["fits"]["B"]["q"], vectors[pole]["A"]["vector"])
        targets[pole] = {"full_cosine": full_cosine, "replayed_full_cosine": replayed_full_cosine,
                         "q_A_v_B_cosine": qa_vb, "q_B_v_A_cosine": qb_va,
                         "candidate": bool(supported and full_cosine <= -.30 and qa_vb < 0 and qb_va < 0)}
    artifacts = {"X": X, "y": y, "A": A, "q": full["q"],
                 "q_A": cross["fits"]["A"]["q"], "q_B": cross["fits"]["B"]["q"],
                 "oof_y": cross["oof_y"], "oof_prediction": cross["oof_prediction"],
                 "full_beta_y": full["beta_y"], "full_beta_A": full["beta_A"]}
    for pole in POLES:
        for subset in ("full", "A", "B"):
            artifacts[f"vector_{pole}_{subset}"] = vectors[pole][subset]["vector"]
    summary = {"n_answers": len(rows), "n_pairs": len(y), "n_questions": 80,
               "formal_e0_pass": False, "injection_test_performed": False,
               "design_columns": DESIGN_COLUMNS, "rank": full["rank"], "condition_number": full["condition_number"],
               "residual_y_energy": full["residual_y_energy"], "replay_checks": replay_checks,
               "folds": cross["folds"], "oof_rho": cross["rho"], "pole_side": cross["pole_side"],
               "bootstrap": bootstrap, "split_direction_cosine": stability,
               "direction_supported": bool(supported), "targets": targets,
               "pair_rows": pairs["pair_rows"],
               "limitations": ["主评委连贯性相关方向，不是纯质量或因果方向", "同提取池几何不能替代实际干预及任务理解验证",
                               "未使用人格评测集、正式 test 或新评分"]}
    return {"summary": summary, "artifacts": artifacts}
