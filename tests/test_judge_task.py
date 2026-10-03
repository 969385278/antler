"""C20：任务理解提示逐字来自规格，独立评分且不调用远端 API。"""
import re
from pathlib import Path

from antler.eval.judge import Judge, TASK_PROMPT


def test_task_prompt_matches_research_spec():
    text = (Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_SPEC.md").read_text(encoding="utf-8")
    section = text.split("**(a′)", 1)[1]
    expected = re.search(r"```\n(.*?)\n```", section, re.DOTALL).group(1)
    assert TASK_PROMPT == expected


def test_task_scores_uses_only_task_prompt(monkeypatch):
    judge = object.__new__(Judge)
    prompts = []

    def score_many(values):
        prompts.extend(values)
        return [90, None]

    monkeypatch.setattr(judge, "score_many", score_many)
    assert judge.task_scores(["问题一", "问题二"], ["回答一", "回答二"]) == [90, None]
    assert prompts == [TASK_PROMPT.format(question=q, answer=a)
                       for q, a in zip(["问题一", "问题二"], ["回答一", "回答二"])]
