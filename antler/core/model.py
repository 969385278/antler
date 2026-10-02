"""加载冻结的模型，以及和模型结构相关的小工具。"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name_or_path: str, dtype: str = "bfloat16", device: str = "auto"):
    """加载模型和分词器，并冻结全部参数（整个项目从不修改模型权重）。"""
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # 生成时左填充

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch_dtype)
    model.to(device).eval()
    freeze(model)
    return model, tok


def freeze(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def get_layers(model):
    """返回 decoder 层列表。Qwen2/Qwen3/Llama 都是 model.model.layers。"""
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"不认识的模型结构：{type(model).__name__}")


def n_layers(model) -> int:
    return len(get_layers(model))


def hidden_size(model) -> int:
    return model.config.hidden_size


def chat_prompt(tok, user: str, system: str | None = None) -> str:
    """把一轮对话渲染成模型的对话模板（到助手开始回答的位置为止）。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def model_device(model) -> torch.device:
    return next(model.parameters()).device
