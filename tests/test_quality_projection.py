"""C18：同范数投影、真实冻结模型注入、按题聚合与停止条件。"""
import copy
from fractions import Fraction

import numpy as np
import pytest
import torch

from antler.core.hooks import Recorder, Steer
from antler.core.model import get_layers
from antler.traits import quality_projection as projection
from antler.traits.data import BIG_FIVE
from test_core import D, model


@pytest.fixture
def traits():
    return {pole: {"questions": [f"{pole} calib {i}" for i in range(10)]}
            for dim in ("C", "N") for pole in BIG_FIVE[dim]}


def make_rows(traits):
    rows = []
    for dim in ("C", "N"):
        for pole, target in zip(("high", "low"), BIG_FIVE[dim]):
            for arm in projection.ARMS:
                for seed in projection.SEEDS:
                    for question in range(10):
                        for sample in range(3):
                            rows.append({"dim": dim, "pole": pole, "target": target, "arm": arm,
                                         "seed": seed, "question_index": question, "sample_index": sample,
                                         "q": traits[target]["questions"][question], "layer": 21,
                                         "coef": projection.COEFS[dim][arm], "axis": "pole", "split": "calib",
                                         "steer_phase": projection.arm_phase(arm), "finish_reason": "eos",
                                         "trait": 20. if arm == "zero" else 50.,
                                         "coherence": {"zero": 95., "original": 80., "projected": 88.}[arm],
                                         "task": {"zero": 95., "original": 80., "projected": 88.}[arm]})
    return rows


def test_projection_preserves_actual_l2_and_signed_single_pole_direction_without_mutation():
    original, q = torch.tensor([-3., 4.]), torch.tensor([1., 0.], dtype=torch.float64)
    original_copy, q_copy = original.clone(), q.clone()
    result = projection.project_vector(original, q)
    assert torch.equal(original, original_copy) and torch.equal(q, q_copy)
    assert torch.equal(result["vector"], torch.tensor([0., 5.]))
    audit = result["audit"]
    assert audit["r"] == pytest.approx(-.6)
    assert audit["retained_norm"] == pytest.approx(.8)
    assert audit["renorm_factor"] == pytest.approx(1.25)
    assert audit["original_actual_norm"] == audit["projected_actual_norm"] == 5.
    assert audit["projected_q_cosine"] == 0
    assert result["vector"].dtype == torch.float32
    opposite_q = projection.project_vector(original, -q)
    assert torch.equal(opposite_q["vector"], result["vector"])
    assert opposite_q["audit"]["r"] == pytest.approx(.6)
    assert not audit["same_trait_effect_established"]


@pytest.mark.parametrize("original,q", [
    ([0., 0.], [1., 0.]), ([1., 2.], [2., 0.]), ([1., 2.], [0., 0.]),
    ([1., 0.], [1., 0.]), ([1., 1e-7], [1., 0.]),
    ([float("nan"), 1.], [1., 0.]), ([1., 1.], [float("inf"), 0.]),
    ([[1., 2.]], [1., 0.]), ([1., 2., 3.], [1., 0.]), ([1, 2], [1., 0.]),
    ([1.], [1.]), ([1., 2.], [1., 0.01]),
])
def test_projection_rejects_invalid_or_unstable_direction(original, q):
    with pytest.raises(ValueError): projection.project_vector(original, q)


def test_projection_checks_float32_conversion_instead_of_returning_infinite_or_zero():
    with pytest.raises(ValueError, match="float32"):
        projection.project_vector(torch.tensor([1e39, 2e39], dtype=torch.float64), torch.tensor([1., 0.]))
    with pytest.raises(ValueError, match="float32"):
        projection.project_vector(torch.tensor([1e-48, 2e-48], dtype=torch.float64), torch.tensor([1., 0.]))


def test_true_decode_steer_applies_projected_vector_with_frozen_weights(model):
    original = torch.zeros(D); original[0], original[1] = -3., 4.
    q = torch.zeros(D, dtype=torch.float64); q[0] = 1.
    projected = projection.project_vector(original, q)["vector"]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    ids = torch.tensor([[3, 5, 7]])
    def execute(vector):
        with torch.no_grad(), Steer(model, {2: vector}, phase="decode") as steer, Recorder(model, [2]) as recorder:
            prefill = model(input_ids=ids, use_cache=True)
            prefill_acts = recorder.acts[2].clone()
            output = model(input_ids=torch.tensor([[9]]), past_key_values=prefill.past_key_values, use_cache=True)
            return prefill.logits.clone(), prefill_acts, output.logits.clone(), recorder.acts[2].clone(), steer.phase_trace
    zero, orig, changed = [execute(vector) for vector in (torch.zeros(D), original, projected)]
    assert torch.equal(orig[0], zero[0]) and torch.equal(changed[0], zero[0])
    assert torch.equal(orig[1], zero[1]) and torch.equal(changed[1], zero[1])
    assert torch.allclose(orig[3] - zero[3], original.expand_as(zero[3]), atol=1e-6)
    assert torch.allclose(changed[3] - zero[3], projected.expand_as(zero[3]), atol=1e-6)
    assert torch.allclose(changed[3] - orig[3], (projected - original).expand_as(zero[3]), atol=1e-6)
    assert not torch.allclose(changed[2], orig[2], atol=1e-5)
    assert [entry["applied"] for entry in changed[4]] == [False, True]
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(not layer._forward_hooks for layer in get_layers(model))


@pytest.mark.parametrize("change", ["missing", "duplicate", "seed0", "seed_bool", "question_float", "test_index",
                                    "sample3", "arm", "target", "question", "coef", "coef_bool", "layer", "axis", "split", "phase"])
def test_fixed_1080_identity_protocol(traits, change):
    rows = make_rows(traits)
    projection.validate_rows(rows, traits)
    if change == "missing": rows.pop()
    elif change == "duplicate": rows[-1] = copy.deepcopy(rows[0])
    else:
        key, value = {"seed0": ("seed", 0), "seed_bool": ("seed", True), "question_float": ("question_index", 0.),
                      "test_index": ("question_index", 10), "sample3": ("sample_index", 3), "arm": ("arm", "extra"),
                      "target": ("target", "wrong"), "question": ("q", "test question"), "coef": ("coef", -.2),
                      "coef_bool": ("coef", False), "layer": ("layer", 16), "axis": ("axis", "bipolar"),
                      "split": ("split", "test"), "phase": ("steer_phase", "decode")}[change]
        rows[0][key] = value
    with pytest.raises(ValueError): projection.validate_rows(rows, traits)


def test_full_twenty_question_scale_is_compatible_but_tail_never_enters_calib(traits):
    rows = make_rows(traits)
    full = {pole: {"questions": item["questions"] + [f"heldout test {i}" for i in range(10)]}
            for pole, item in traits.items()}
    projection.validate_rows(rows, full)
    # 验证器不访问 test 文本；全部统计仍只接收已核验的前十题身份。
    class UnreadTail(list):
        def __getitem__(self, item):
            if isinstance(item, int) and item >= 10:
                raise AssertionError("不得访问 test 题")
            if isinstance(item, slice) and (item.stop is None or item.stop > 10):
                raise AssertionError("不得切片 test 题")
            return super().__getitem__(item)
    for item in full.values(): item["questions"] = UnreadTail(item["questions"])
    projection.validate_rows(rows, full)
    rows[0]["question_index"] = 10
    with pytest.raises(ValueError, match="固定 calib"): projection.validate_rows(rows, full)


def test_full_metrics_candidates_and_method_benefit_are_separate_from_formal_acceptance(traits):
    result = projection.analyze(make_rows(traits), traits)
    assert result["n_samples"] == 1080 and len(result["conditions"]) == 8
    assert len(result["candidates"]) == 4
    assert result["quality_comparisons"]["n_tests"] == 8
    assert len(result["quality_comparisons"]["tests"]) == 8
    assert all(test["adjusted_p"] == pytest.approx(2 / 1024) for test in result["quality_comparisons"]["tests"])
    assert all(test["quality_improvement_supported"] for test in result["quality_comparisons"]["tests"])
    for condition in result["conditions"]:
        metric = condition["pooled"]["versus_zero"]
        assert set(metric) == {"trait", "coherence", "task"}
        assert metric["trait"]["mean"] == 30
        assert metric["trait"]["n_questions"] == 10
        assert len(condition["per_seed"]) == 3
        assert all(value == 0 for value in condition["seed_stddev"]["versus_zero"].values())
    assert all(candidate["pooled"]["numeric_criteria_met"] == (candidate["arm"] == "projected") for candidate in result["candidates"])
    assert all(item["investment_criteria_met"] and not item["stop_current_dimension_projection"] for item in result["method_decisions"])
    assert {item["dim"]: item["target_pole"] for item in result["method_decisions"]} == {"C": "low", "N": "high"}
    assert not result["formal_e0_pass"] and not result["capability_evaluated"]
    assert "sensitivity" not in result


def test_calib_candidate_without_five_point_method_benefit_stops_projection(traits):
    rows = make_rows(traits)
    for row in rows:
        if row["arm"] == "original": row["coherence"] = row["task"] = 87.
        if row["arm"] == "projected": row["coherence"] = row["task"] = 90.
    result = projection.analyze(rows, traits)
    for item in result["method_decisions"]:
        assert item["projected_calib_candidate"]
        assert not item["investment_criteria_met"] and item["stop_current_dimension_projection"]
        assert "projected_full_calib_criteria" not in item["failed_checks"]
        assert len(item["failed_checks"]) == 2


def test_quality_improvement_without_trait_gain_is_not_method_success(traits):
    rows = make_rows(traits)
    for row in rows:
        if row["arm"] == "projected" and row["dim"] == "C": row["trait"] = 35.
    result = projection.analyze(rows, traits)
    decisions = {item["dim"]: item for item in result["method_decisions"]}
    assert not decisions["C"]["projected_calib_candidate"]
    assert decisions["C"]["failed_checks"] == ["projected_full_calib_criteria"]
    assert decisions["C"]["stop_current_dimension_projection"]
    assert decisions["N"]["investment_criteria_met"]


def test_nontarget_task_failure_cannot_be_hidden_by_target_improvement(traits):
    rows = make_rows(traits)
    for row in rows:
        if row["dim"] == "C" and row["pole"] == "high" and row["arm"] == "projected": row["task"] = 84.
    result = projection.analyze(rows, traits)
    candidate = next(item for item in result["candidates"] if item["dim"] == "C" and item["arm"] == "projected")
    assert candidate["pooled"]["failed_checks"] == ["high_task_drop_le10"]
    assert next(item for item in result["method_decisions"] if item["dim"] == "C")["stop_current_dimension_projection"]


def test_nine_repeated_responses_still_produce_ten_question_interval_and_seed_sd(traits):
    rows = make_rows(traits)
    for row in rows:
        if row["arm"] != "zero":
            gain = 0 if row["question_index"] < 5 else 50
            row["trait"] = 20. + gain + (row["sample_index"] - 1) * 3 + (row["seed"] - 2) * 2
    result = projection.analyze(rows, traits)
    for condition in result["conditions"]:
        estimate = condition["pooled"]["versus_zero"]["trait"]
        assert estimate["mean"] == 25.
        assert estimate["ci_low"] == 10. and estimate["ci_high"] == 40.
        assert estimate["n_questions"] == 10
        assert condition["seed_stddev"]["versus_zero"]["trait"] == 2.
    assert all("target_trait_ci_low_gt10" in item["pooled"]["failed_checks"] for item in result["candidates"])


def test_exact_cutoffs_and_effect_difference_are_reported_without_lowering_thresholds(traits):
    rows = make_rows(traits)
    for row in rows:
        row["coherence"] = row["task"] = {"zero": 85., "original": 70., "projected": 75.}[row["arm"]]
        if row["arm"] == "projected": row["trait"] = 40.
    result = projection.analyze(rows, traits)
    assert all(item["investment_criteria_met"] for item in result["method_decisions"])
    assert all(not item["target_trait_versus_original"]["ci_within_minus5_plus5"] for item in result["method_decisions"])
    assert all(item["target_trait_versus_original"]["mean"] == -10 for item in result["method_decisions"])
    for item in result["candidates"]:
        if item["arm"] == "projected":
            assert item["pooled"]["numeric_criteria_met"]
            assert len(item["pooled"]["quality_boundary_uncertainty"]) == 4


@pytest.mark.parametrize("metric,value", [("trait", None), ("coherence", float("nan")), ("task", 101), ("task", True)])
def test_missing_or_invalid_scores_fail_without_filtering(traits, metric, value):
    rows = make_rows(traits); rows[0][metric] = value
    with pytest.raises(ValueError, match="三项评分全部有效"): projection.analyze(rows, traits)


def test_exact_sign_flip_preserves_ties_and_is_two_sided():
    assert projection.exact_sign_flip([0.] * 10) == 1.
    assert projection.exact_sign_flip([Fraction(1, 3)] * 10) == 2 / 1024
    assert projection.exact_sign_flip([Fraction(-1, 3)] * 10) == 2 / 1024
    assert projection.exact_sign_flip([1.] * 5 + [-1.] * 5) == 1.
    with pytest.raises(ValueError): projection.exact_sign_flip([1.] * 9)
