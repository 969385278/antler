"""把下载好的 TRAIT 数据转换成实验用的四选一格式：data/trait_big5_mcq.jsonl

用法：python scripts/prepare_trait.py --src data/hf/TRAIT
只保留大五，四个选项随机打乱（避免模型偏爱某个位置），is_high 标记哪些选项代表高特质。
"""
import argparse
import glob
import json
import random

import pandas as pd

BIG5 = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}


def read_any(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf/TRAIT")
    ap.add_argument("--out", default="data/trait_big5_mcq.jsonl")
    args = ap.parse_args()
    files = [f for f in glob.glob(f"{args.src}/**/*", recursive=True)
             if f.endswith((".parquet", ".json", ".jsonl", ".csv"))]
    print("找到文件：", files)
    df = pd.concat([read_any(f) for f in files], ignore_index=True)
    print("列：", list(df.columns), "  总数：", len(df))
    print("特质分布：\n", df["personality"].value_counts())

    rng = random.Random(0)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            if str(r["personality"]).lower() not in BIG5:
                continue
            opts = [(r["response_high1"], 1), (r["response_high2"], 1), (r["response_low1"], 0), (r["response_low2"], 0)]
            rng.shuffle(opts)
            f.write(json.dumps({"trait": r["personality"], "question": r["question"],
                                "options": [o for o, _ in opts], "is_high": [h for _, h in opts]},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"已写入 {n} 条到 {args.out}")


if __name__ == "__main__":
    main()
