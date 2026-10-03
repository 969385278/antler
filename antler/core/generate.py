"""批量生成文本。"""
from __future__ import annotations

import torch

from .hooks import STEER_PHASES, Steer, no_steer
from .model import model_device


@torch.no_grad()
def generate(model, tok, prompts: list[str], *, steer: dict[int, torch.Tensor] | None = None,
             max_new_tokens: int = 256, temperature: float = 1.0, top_p: float = 1.0,
             batch_size: int = 16, seed: int | None = None,
             return_details: bool = False, return_token_ids: bool = False,
             steer_phase: str = "all", phase_trace_out: list | None = None) -> list[str] | list[dict]:
    """prompts 是已经套好对话模板的字符串。返回每条 prompt 生成的回答文本。

    temperature=0 表示贪心解码。默认整个生成过程注入；steer_phase 可选 prefill/decode。
    非默认阶段仅支持有 KV cache 的普通单束生成，不支持分块提示或辅助解码。
    return_details=True 时返回 text、n_generated_tokens、finish_reason。
    token 数包括首个 EOS，不包括提示或停止后的批补齐；恰好在长度上限输出 EOS 也记为 eos。
    return_token_ids=True 时额外保存实际生成 token，必须同时开启 return_details。
    phase_trace_out 若为列表，每批完成后追加阶段标量记录；不改变返回字段，all 的 trace 为空。
    """
    if return_token_ids and not return_details:
        raise ValueError("return_token_ids 需要同时开启 return_details")
    if steer_phase not in STEER_PHASES:
        raise ValueError("steer_phase 必须是 all、prefill 或 decode")
    if phase_trace_out is not None and not isinstance(phase_trace_out, list):
        raise ValueError("phase_trace_out 必须是列表或 None")
    if seed is not None:
        torch.manual_seed(seed)
    tok.padding_side = "left"
    device = model_device(model)
    outs = []
    # 使用模型实际生成配置；tokenizer 的 EOS 不一定是 generate 的停止条件。
    generation_config = getattr(model, "generation_config", getattr(model, "config", None))
    if steer_phase != "all":
        use_cache = getattr(generation_config, "use_cache", None)
        if use_cache is None:
            use_cache = getattr(getattr(model, "config", None), "use_cache", None)
        if use_cache is not True:
            raise ValueError("prefill/decode 阶段注入需要 use_cache=True")
        if getattr(generation_config, "prefill_chunk_size", None) is not None:
            raise ValueError("阶段注入暂不支持分块 prefill")
        if (getattr(generation_config, "num_beams", None) not in (None, 1)
                or getattr(generation_config, "num_beam_groups", None) not in (None, 1)
                or getattr(generation_config, "is_assistant", False)
                or getattr(generation_config, "prompt_lookup_num_tokens", None) is not None
                or getattr(generation_config, "assistant_early_exit", None) is not None):
            raise ValueError("阶段注入仅支持普通单束生成，不支持辅助解码")
        if (getattr(generation_config, "cache_implementation", None) not in (None, "dynamic")
                or getattr(getattr(model, "config", None), "is_encoder_decoder", False)
                or hasattr(model, "_orig_mod")):
            raise ValueError("阶段注入仅支持未编译的 decoder 模型与动态 KV cache")
    eos_ids = getattr(generation_config, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids or [])
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if steer_phase != "all":
            # 阶段观察 hook 需要逐次执行；只在新模式下禁用生成过程的自动编译。
            kwargs.update(use_cache=True, disable_compile=True)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        context = (Steer(model, steer) if steer else no_steer()) if steer_phase == "all" else (
            Steer(model, steer or {}, phase=steer_phase))
        with context:
            gen = model.generate(**enc, **kwargs)
        if phase_trace_out is not None:
            phase_trace_out.append({"batch_start": i, "trace": [] if steer_phase == "all" else (
                [dict(entry) for entry in context.phase_trace])})
        sequences = gen.sequences if hasattr(gen, "sequences") else gen
        new_tokens = sequences[:, enc["input_ids"].shape[1]:]
        if not return_details:
            outs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
            continue
        for token_ids in new_tokens.tolist():
            # 先找 EOS，再截掉批补齐；pad==eos 时不能先按 pad 去尾。
            eos_at = next((j for j, token in enumerate(token_ids) if token in eos_ids), None)
            count = len(token_ids) if eos_at is None else eos_at + 1
            text = tok.batch_decode([token_ids[:count]], skip_special_tokens=True)[0]
            outs.append({"text": text, "n_generated_tokens": count,
                         "finish_reason": "length" if eos_at is None else "eos",
                         **({"token_ids": token_ids[:count]} if return_token_ids else {})})
    return outs
