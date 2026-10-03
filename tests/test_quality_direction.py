"""C18：已知机制、未见题隔离及整题重抽；只用 CPU 合成数据。"""
import copy
import hashlib

import numpy as np
import pytest

from antler.traits import quality_direction as quality


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(712)
    rows, acts, tokens = [], [], []
    q = np.array([1., 0., 0., 0., 0., 0.])
    for pole_index, pole in enumerate(quality.POLES):
        source_index = 0
        for pair in range(5):
            for question in range(20):
                for rollout in range(2):
                    for side in quality.SIDES:
                        score = (70 if side == "pos" else 30) + rng.uniform(-15, 15)
                        coh = rng.uniform(20, 95)
                        token = int(rng.integers(20, 300))
                        direction = -1 if pole in quality.TARGETS else 1
                        offset = direction * (100 if side == "pos" else 0) * q
                        activation = offset + coh * q
                        activation[1 + pole_index] += score * .9
                        activation[5] += np.log1p(token) * 40
                        activation += rng.normal(0, .001, 6)
                        rows.append({"pole": pole, "source_index": source_index, "pair_index": pair,
                                     "question": f"{pole} question {question}", "rollout": rollout,
                                     "side": side, "score": score, "coherence": coh})
                        acts.append(activation)
                        tokens.append(token)
                        source_index += 1
    return rows, np.asarray(acts), np.asarray(tokens), q


def references(sample):
    rows, acts, _, _ = sample
    return {pole: item["full"]["vector"] for pole, item in quality.reconstruct_paired_vectors(rows, acts).items()}


def test_folds_only_depend_on_identity_and_keep_all_twenty_answers_together(sample):
    rows, _, _, _ = sample
    expected = quality.quality_question_folds(rows)
    shuffled = copy.deepcopy(rows)
    np.random.default_rng(91).shuffle(shuffled)
    for row in shuffled:
        row["score"], row["coherence"] = -999, float("nan")
    actual = quality.quality_question_folds(shuffled)
    mapping = lambda items: {(r["pole"], r["source_index"]): r["fold"] for r in items}
    assert mapping(expected) == mapping(actual)
    for pole in quality.POLES:
        questions = {r["question"] for r in rows if r["pole"] == pole}
        ordered = sorted(questions, key=lambda question: hashlib.sha256(
            f"quality-direction-v1\n{pole}\n{question}".encode()).hexdigest())
        for i, question in enumerate(ordered):
            part = [r for r in expected if r["pole"] == pole and r["question"] == question]
            assert len(part) == 20
            assert {r["fold"] for r in part} == {"A" if i % 2 == 0 else "B"}


@pytest.mark.parametrize("change", ["missing", "duplicate", "fold", "source", "pair", "rollout"])
def test_reject_invalid_or_changed_identity(sample, change):
    rows = copy.deepcopy(sample[0])
    if change == "missing": rows.pop()
    if change == "duplicate": rows[0] = rows[1].copy()
    if change == "fold": rows = quality.quality_question_folds(rows); rows[0]["fold"] = "wrong"
    if change == "source": rows[0]["source_index"] = True
    if change == "pair": rows[0]["pair_index"] = 5
    if change == "rollout": rows[0]["rollout"] = 2
    with pytest.raises(ValueError): quality.quality_question_folds(rows)


def test_pair_difference_preserves_rollout_sign_and_pole_specific_trait_controls(sample):
    rows, acts, tokens, _ = sample
    result = quality.prepare_quality_pairs(rows, acts, tokens, 2.)
    assert result["A"].shape == (800, 6)
    assert result["X"].shape == (800, 13)
    assert np.linalg.matrix_rank(result["X"]) == 13
    for index, pair in enumerate(result["pair_rows"]):
        a, b = pair["row0"], pair["row1"]
        assert rows[a]["rollout"] == 0 and rows[b]["rollout"] == 1
        assert result["y"][index] == rows[b]["coherence"] - rows[a]["coherence"]
        assert np.allclose(result["A"][index], (acts[b] - acts[a]) / 2)
        slope = result["X"][index, 8:12]
        assert np.count_nonzero(slope) == 1
        assert slope[quality.POLES.index(pair["pole"])] == rows[b]["score"] - rows[a]["score"]
    assert np.any(result["y"] < 0) and np.any(result["y"] > 0)


def test_known_quality_direction_recovered_after_large_trait_and_length_confound(sample):
    rows, acts, tokens, q = sample
    pairs = quality.prepare_quality_pairs(rows, acts, tokens, 2.)
    fitted = quality.fit_quality_direction(pairs["X"], pairs["y"], pairs["A"])
    assert fitted["q"] @ q > .999999
    assert np.allclose(pairs["X"].T @ fitted["residual_y"], 0, atol=1e-7)
    assert fitted["calibration_slope"] == pytest.approx(2., rel=1e-5)
    # 同一质量关联不应被纯 nuisance 平移解释成另一方向。
    nuisance = np.arange(13 * 6).reshape(13, 6) * 9.
    contaminated = pairs["A"] + pairs["X"] @ nuisance
    again = quality.fit_quality_direction(pairs["X"], pairs["y"], contaminated)
    assert np.allclose(fitted["q"], again["q"], atol=1e-10)


def test_heldout_labels_never_affect_training_direction_or_heldout_predictions(sample):
    rows, acts, tokens, _ = sample
    pairs = quality.prepare_quality_pairs(rows, acts, tokens, 1.)
    original = quality.crossfit_quality_direction(pairs["X"], pairs["y"], pairs["A"], pairs["pair_rows"])
    heldout = np.asarray([r["fold"] == "B" for r in pairs["pair_rows"]])
    changed_y = pairs["y"].copy()
    changed_y[heldout] = np.random.default_rng(5).normal(1000, 100, heldout.sum())
    changed = quality.crossfit_quality_direction(pairs["X"], changed_y, pairs["A"], pairs["pair_rows"])
    assert np.array_equal(original["fits"]["A"]["q"], changed["fits"]["A"]["q"])
    assert np.array_equal(original["oof_prediction"][heldout], changed["oof_prediction"][heldout])
    assert not np.allclose(original["oof_y"][heldout], changed["oof_y"][heldout])
    assert original["rho"] > .999
    assert len(original["pole_side"]) == 8


@pytest.mark.parametrize("kind", ["rank", "quality_zero", "activation_zero", "nonfinite"])
def test_invalid_fits_fail_instead_of_changing_estimator(sample, kind):
    rows, acts, tokens, _ = sample
    pairs = quality.prepare_quality_pairs(rows, acts, tokens, 1.)
    X, y, A = [pairs[key].copy() for key in ("X", "y", "A")]
    if kind == "rank": X[:, 12] = X[:, 11]
    if kind == "quality_zero": y = X @ np.arange(13)
    if kind == "activation_zero": A[:] = 0
    if kind == "nonfinite": A[0, 0] = np.nan
    with pytest.raises(ValueError): quality.fit_quality_direction(X, y, A)


@pytest.mark.parametrize("kind", ["tokens", "missing_score", "coherence", "scale", "activation"])
def test_pair_inputs_cannot_silently_drop_or_replace_rows(sample, kind):
    rows, acts, tokens, _ = copy.deepcopy(sample)
    scale = 1.
    if kind == "tokens": tokens[0] = 0
    if kind == "missing_score": rows[0]["score"] = None
    if kind == "coherence": rows[0]["coherence"] = float("inf")
    if kind == "scale": scale = -1.
    if kind == "activation": acts = acts[:-1]
    with pytest.raises(ValueError): quality.prepare_quality_pairs(rows, acts, tokens, scale)


def test_paired_reconstruction_uses_exact_boundary_and_positive_low_pole_sign(sample):
    rows, acts, _, q = copy.deepcopy(sample)
    rows[0]["score"], rows[1]["score"] = 50., 49.
    rows[4]["score"], rows[5]["score"] = 50., 50.
    vectors = quality.reconstruct_paired_vectors(rows, acts)
    assert vectors["dependable"]["full"]["n_pairs"] == 199
    assert [0, 1] in vectors["dependable"]["full"]["source_pairs"]
    assert [4, 5] not in vectors["dependable"]["full"]["source_pairs"]
    for pole in quality.POLES:
        assert sum(vectors[pole][fold]["n_pairs"] for fold in quality.FOLDS) == vectors[pole]["full"]["n_pairs"]
    assert vectors["careless"]["full"]["vector"] @ q < 0
    assert vectors["nervous"]["full"]["vector"] @ q < 0


def test_bootstrap_samples_ten_pairs_as_one_question_with_fixed_fold_and_pole(sample):
    rows, acts, tokens, _ = sample
    pairs = quality.prepare_quality_pairs(rows, acts, tokens, 1.)
    weights = quality.bootstrap_question_weights(pairs["pair_rows"], np.random.default_rng(0))
    assert np.array_equal(weights, quality.bootstrap_question_weights(pairs["pair_rows"], np.random.default_rng(0)))
    assert np.any(weights == 0) and np.any(weights > 1)
    for fold in quality.FOLDS:
        for pole in quality.POLES:
            mask = [r["fold"] == fold and r["pole"] == pole for r in pairs["pair_rows"]]
            assert weights[mask].sum() == 100
            questions = {r["question"] for r in pairs["pair_rows"] if r["fold"] == fold and r["pole"] == pole}
            for question in questions:
                group = [r["pole"] == pole and r["question"] == question for r in pairs["pair_rows"]]
                assert sum(group) == 10 and len(set(weights[group])) == 1


def test_weighted_refit_matches_literal_repeated_questions_and_bootstrap_percentiles(sample):
    rows, acts, tokens, _ = sample
    pairs = quality.prepare_quality_pairs(rows, acts, tokens, 1.)
    rng, literal_rhos = np.random.default_rng(0), []
    for _ in range(4):
        weights = quality.bootstrap_question_weights(pairs["pair_rows"], rng)
        repeated = np.repeat(np.arange(800), weights)
        literal_rows = [pairs["pair_rows"][i] for i in repeated]
        literal = quality.crossfit_quality_direction(pairs["X"][repeated], pairs["y"][repeated],
                                                     pairs["A"][repeated], literal_rows)
        weighted = quality.crossfit_quality_direction(pairs["X"], pairs["y"], pairs["A"],
                                                      pairs["pair_rows"], weights=weights)
        for fold in quality.FOLDS:
            for key in ("q", "beta_y", "beta_A", "calibration_slope"):
                assert np.allclose(literal["fits"][fold][key], weighted["fits"][fold][key], atol=1e-9)
        assert literal["rho"] == pytest.approx(weighted["rho"], abs=1e-12)
        literal_rhos.append(literal["rho"])
    result = quality.bootstrap_quality_direction(pairs["X"], pairs["y"], pairs["A"], pairs["pair_rows"], bootstrap_samples=4)
    assert result["rho_values"] == pytest.approx(literal_rhos, abs=1e-12)
    assert result["rho_ci95"] == pytest.approx(np.quantile(literal_rhos, [.025, .975]), abs=1e-12)


def test_bootstrap_failure_is_preserved_and_stops_without_redrawing(sample, monkeypatch):
    rows, acts, tokens, _ = sample
    pairs = quality.prepare_quality_pairs(rows, acts, tokens, 1.)
    actual, calls = quality.crossfit_quality_direction, []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise ValueError("设计矩阵秩不足：12/13")
        return actual(*args, **kwargs)
    monkeypatch.setattr(quality, "crossfit_quality_direction", fail_second)
    result = quality.bootstrap_quality_direction(pairs["X"], pairs["y"], pairs["A"], pairs["pair_rows"], bootstrap_samples=10)
    assert len(calls) == 2
    assert not result["complete"] and result["rho_ci95"] is None
    assert result["attempted"] == 2 and result["rho_values"][1] is None
    assert result["failures"] == [{"replicate": 1, "error": "设计矩阵秩不足：12/13"}]


def test_complete_known_signal_supports_negative_targets_and_keeps_positive_target_out(sample):
    rows, acts, tokens, _ = sample
    result = quality.analyze_quality_direction(rows, acts, tokens, 1., references(sample), bootstrap_samples=12)
    summary = result["summary"]
    assert summary["direction_supported"]
    assert summary["bootstrap"]["rho_ci95"][0] > .99
    assert summary["split_direction_cosine"] > .99
    assert all(summary["targets"][pole]["candidate"] for pole in quality.TARGETS)
    # 翻转的是 nervous 的人格基线差，不能使 q 为匹配目标而跟着翻转。
    changed = acts.copy()
    for i, row in enumerate(rows):
        if row["pole"] == "nervous" and row["side"] == "pos": changed[i, 0] += 200
    ref = references((rows, changed, tokens, sample[3]))
    alternate = quality.analyze_quality_direction(rows, changed, tokens, 1., ref, bootstrap_samples=4)
    assert alternate["summary"]["targets"]["nervous"]["full_cosine"] > .3
    assert not alternate["summary"]["targets"]["nervous"]["candidate"]
    assert alternate["summary"]["targets"]["careless"]["candidate"]
    assert result["artifacts"]["q"] @ alternate["artifacts"]["q"] > .999999


def test_reference_mismatch_stops_before_bootstrap(sample, monkeypatch):
    rows, acts, tokens, _ = sample
    ref = references(sample)
    ref["careless"] = ref["careless"] * 1.20
    def forbidden(*args, **kwargs): raise AssertionError("不一致时不得继续解释统计")
    monkeypatch.setattr(quality, "bootstrap_quality_direction", forbidden)
    with pytest.raises(ValueError, match="重放与既有 paired 向量不一致"):
        quality.analyze_quality_direction(rows, acts, tokens, 1., ref, bootstrap_samples=2)


def test_full_target_cosine_uses_saved_actual_vector_not_replay_approximation(sample):
    rows, acts, tokens, _ = sample
    ref = references(sample)
    ref["careless"] = ref["careless"].copy()
    ref["careless"][1] += .5
    result = quality.analyze_quality_direction(rows, acts, tokens, 1., ref, bootstrap_samples=2)
    expected = result["artifacts"]["q"] @ ref["careless"] / np.linalg.norm(ref["careless"])
    target = result["summary"]["targets"]["careless"]
    assert target["full_cosine"] == pytest.approx(expected, abs=1e-12)
    assert abs(target["full_cosine"] - target["replayed_full_cosine"]) > 1e-6


def test_average_ranks_with_ties_and_undefined_constant_group():
    assert quality.spearman_correlation([1, 1, 3, 4], [5, 5, 2, 1]) == pytest.approx(-1.)
    assert quality.spearman_correlation([1, 1], [0, 1]) is None
