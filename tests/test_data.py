"""数据检查：不需要 PyTorch。"""
from antler.eval.judge import parse_score
from antler.traits.data import BIG_FIVE, DIMS, POLES, check_no_overlap, load_personality_bench, load_trait


def test_all_poles_exist_and_shapes():
    assert len(POLES) == 10
    for p in POLES:
        for split in ("extract", "eval"):
            d = load_trait(p, split)
            assert len(d["instruction"]) == 5 and len(d["questions"]) == 20
            assert "{question}" in d["eval_prompt"] and "{answer}" in d["eval_prompt"]


def test_extract_and_eval_do_not_overlap():
    assert all(v == 0 for v in check_no_overlap().values())


def test_personality_bench_loads():
    for k in DIMS:
        assert len(load_personality_bench(k)) >= 85


def test_parse_score():
    assert parse_score("85") == 85
    assert parse_score(" 42.5\n") == 42.5
    assert parse_score("REFUSAL") is None
    assert parse_score("n/a") is None
    assert parse_score("150") is None
