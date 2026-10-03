"""C20：生成停止信息不混入提示 token 或批补齐，使用假输出和随机 CPU 微型模型。"""
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from antler.core.generate import generate
from test_core import model


class Tokenizer:
    eos_token_id = 2

    def __init__(self, pad=0):
        self.pad_token_id = pad
        self.padding_side = "right"

    def __call__(self, prompts, **kwargs):
        assert self.padding_side == "left"
        values = [[int(t) for t in prompt.split()] for prompt in prompts]
        width = max(map(len, values))
        return BatchEncoding({
            "input_ids": torch.tensor([[self.pad_token_id] * (width - len(v)) + v for v in values]),
            "attention_mask": torch.tensor([[0] * (width - len(v)) + [1] * len(v) for v in values]),
        })

    def batch_decode(self, tokens, **kwargs):
        return [" ".join(str(t) for t in row if t not in {self.pad_token_id, 2, 3})
                for row in (tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens)]


class FakeModel(torch.nn.Module):
    def __init__(self, rows, eos, return_object=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.rows = rows
        self.return_object = return_object

    def generate(self, input_ids, **kwargs):
        assert input_ids.device.type == "cpu"
        sequences = torch.cat((input_ids, torch.tensor(self.rows)), dim=1)
        return SimpleNamespace(sequences=sequences) if self.return_object else sequences


@pytest.mark.parametrize("pad", [0, 2])
@pytest.mark.parametrize("return_object", [False, True])
def test_details_stop_at_first_eos_and_ignore_left_padding(pad, return_object):
    model = FakeModel([[12, 2, pad, pad], [13, 14, 15, 3], [12, 13, 14, 15]], [2, 3], return_object)
    result = generate(model, Tokenizer(pad), ["2", "2 8 9", "2 8"],
                      max_new_tokens=4, return_details=True)
    assert result == [
        {"text": "12", "n_generated_tokens": 2, "finish_reason": "eos"},
        {"text": "13 14 15", "n_generated_tokens": 4, "finish_reason": "eos"},
        {"text": "12 13 14 15", "n_generated_tokens": 4, "finish_reason": "length"},
    ]


def test_immediate_eos_and_legacy_text_return():
    model = FakeModel([[2, 0, 0]], 2)
    assert generate(model, Tokenizer(), ["8"]) == [""]
    assert generate(model, Tokenizer(), ["8"], return_details=True) == [
        {"text": "", "n_generated_tokens": 1, "finish_reason": "eos"}]


def test_tokenizer_eos_is_not_used_when_model_has_no_eos():
    result = generate(FakeModel([[2, 0, 0]], None), Tokenizer(), ["8"], return_details=True)
    # 模型未配置停止 token 时，EOS 样式的 id 和 pad 都可能是实际生成内容。
    assert result[0]["finish_reason"] == "length" and result[0]["n_generated_tokens"] == 3


def test_tiny_cpu_model_details_match_legacy_and_batch_order(model, monkeypatch):
    monkeypatch.setattr(model.generation_config, "eos_token_id", None)
    prompts = ["4 5", "6", "7 8 9"]
    kw = dict(max_new_tokens=3, temperature=0, batch_size=2, seed=0)
    legacy = generate(model, Tokenizer(), prompts, **kw)
    details = generate(model, Tokenizer(), prompts, return_details=True, **kw)
    assert legacy == [row["text"] for row in details]
    assert all(row["finish_reason"] == "length" and row["n_generated_tokens"] == 3 for row in details)
    assert all(not p.requires_grad for p in model.parameters())


def test_token_ids_keep_first_eos_and_remove_padding():
    result = generate(FakeModel([[12, 3, 2, 2]], [2, 3]), Tokenizer(2), ["8"],
                      max_new_tokens=4, return_details=True, return_token_ids=True)
    assert result[0]["token_ids"] == [12, 3]
    assert result[0]["n_generated_tokens"] == 2
    with pytest.raises(ValueError, match="return_details"):
        generate(FakeModel([[2]], 2), Tokenizer(), ["8"], return_token_ids=True)


def test_tiny_model_token_ids_reproduce_text(model, monkeypatch):
    monkeypatch.setattr(model.generation_config, "eos_token_id", None)
    result = generate(model, Tokenizer(), ["4 5"], max_new_tokens=3, temperature=0,
                      return_details=True, return_token_ids=True)[0]
    assert Tokenizer().batch_decode([result["token_ids"]])[0] == result["text"]
    assert len(result["token_ids"]) == 3


def test_sampling_seed_reproduces_tokens_at_fixed_batch_size(model, monkeypatch):
    monkeypatch.setattr(model.generation_config, "eos_token_id", None)
    kw = dict(max_new_tokens=8, temperature=1., batch_size=2, return_details=True, return_token_ids=True)
    first = generate(model, Tokenizer(), ["4 5", "6"], seed=17, **kw)
    assert first == generate(model, Tokenizer(), ["4 5", "6"], seed=17, **kw)
    assert first != generate(model, Tokenizer(), ["4 5", "6"], seed=18, **kw)
