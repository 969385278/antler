"""C6：确认低极使用独立题库和评分提示，并保持注入符号、默认高极行为。"""
import importlib
import json

import pytest
import torch

from test_core import model  # 复用 CPU 随机微型模型，不下载权重
from antler.core.hooks import Steer, no_steer
from antler.traits.data import BIG_FIVE, DIMS, load_trait

scoring = importlib.import_module("antler.eval.trait_score")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)


@pytest.mark.parametrize("dim", ["C", "N"])
@pytest.mark.parametrize("pole", [None, "high", "low"])
def test_pole_selects_questions_prompt_and_keeps_steering(model, monkeypatch, dim, pole):
    selected = pole or "high"
    trait_name = BIG_FIVE[dim][0 if selected == "high" else 1]
    data = load_trait(trait_name, "eval")
    questions = data["questions"][:2] * 2
    injection = torch.zeros(model.config.hidden_size)
    injection[DIMS.index(dim)] = -0.2
    steer = {1: injection}
    baseline = model(torch.tensor([[1, 2, 3]])).logits.detach()
    calls = []

    def generate(m, tok, prompts, **kwargs):
        assert kwargs["steer"] is steer
        assert kwargs["max_new_tokens"] == 8 and kwargs["seed"] == 7
        assert [json.loads(p)[-1]["content"] for p in prompts] == questions
        assert all(json.loads(p)[0] == {"role": "system", "content": "测试系统提示"} for p in prompts)
        with torch.no_grad(), (Steer(m, steer) if steer else no_steer()):
            changed = m(torch.tensor([[1, 2, 3]])).logits
        assert not torch.allclose(changed, baseline)
        return [f"回答{i}" for i in range(len(prompts))]

    class Judge:
        def trait_scores(self, prompt, qs, answers):
            calls.append("trait")
            assert prompt == data["eval_prompt"]
            assert qs == questions and answers == [f"回答{i}" for i in range(4)]
            return [20, None, 40, 60]

        def coherence_scores(self, qs, answers):
            calls.append("coherence")
            assert qs == questions
            return [80, 90, None, 100]

    monkeypatch.setattr(scoring, "generate", generate)
    kwargs = {} if pole is None else {"pole": pole}
    result = scoring.trait_score(model, Tokenizer(), Judge(), dim, steer=steer,
                                 n_questions=2, n_samples=2, max_new_tokens=8, seed=7,
                                 system="测试系统提示", **kwargs)
    assert result["pole"] == selected and result["trait_name"] == trait_name
    assert result["dim"] == dim and result["trait"] == 40 and result["n_valid"] == 3
    assert result["coherence"] == 90 and calls == ["trait", "coherence"]
    assert [s["q"] for s in result["samples"]] == questions
    assert result["samples"][1]["trait"] is None
    assert torch.allclose(model(torch.tensor([[1, 2, 3]])).logits, baseline)
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize("pole", ["both", "careless", "invalid"])
def test_invalid_pole_fails_before_generation(monkeypatch, pole):
    def unexpected(*args, **kwargs):
        pytest.fail("非法极点不应加载题库或生成回答")

    monkeypatch.setattr(scoring, "load_trait", unexpected)
    monkeypatch.setattr(scoring, "generate", unexpected)
    with pytest.raises(ValueError, match="high 或 low"):
        scoring.trait_score(None, None, None, "C", pole=pole)
