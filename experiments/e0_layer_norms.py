"""E0-a：计算各层激活的平均范数 s_ℓ（所有注入系数的“刻度单位”）。

参考文本优先用 UltraChat（scripts/download_on_server.sh 下载到 data/hf/ultrachat_200k），
没有的话退而使用 PersonalityBench 的情境题（仅供调试）。
"""
import glob

import torch

from _common import dump, parse_args, setup
from antler.core.activations import layer_norms
from antler.core.model import chat_prompt
from antler.traits.data import DIMS, load_personality_bench


def reference_texts(tok, n=300):
    files = glob.glob("data/hf/ultrachat_200k/**/*.parquet", recursive=True)
    if files:
        import pandas as pd
        df = pd.read_parquet(files[0])
        prompts = df["prompt"].tolist()[:n]
        print(f"[参考文本] UltraChat {len(prompts)} 条")
    else:
        prompts = [q for k in DIMS for q in load_personality_bench(k)][:n]
        print(f"[参考文本] 没找到 UltraChat，用 PersonalityBench {len(prompts)} 条代替（仅供调试）")
    return [chat_prompt(tok, p) for p in prompts]


def main():
    args = parse_args()
    cfg, model, tok, _ = setup(args, need_judge=False)
    s = layer_norms(model, tok, reference_texts(tok, cfg.get("ref_n", 300)), batch_size=cfg.get("batch_size", 8))
    torch.save(s, cfg["out_path"] / "layer_norms.pt")
    dump({"layer_norms": s.tolist()}, cfg["out_path"] / "layer_norms.json")
    for l, v in enumerate(s.tolist()):
        print(f"  layer {l:2d}: s = {v:8.2f}")


if __name__ == "__main__":
    main()
