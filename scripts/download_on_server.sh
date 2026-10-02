#!/usr/bin/env bash
# 在 GPU 服务器上运行：下载模型和 HuggingFace 上的数据集
# 用法（在项目根目录）：
#   export HF_TOKEN=hf_xxx        # TRAIT 需要：先在网页上同意使用条件，再填你的只读 token
#   bash scripts/download_on_server.sh
# 模型会下载到 models/，数据到 data/hf/（都已在 .gitignore 中）
set -e

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # 默认：项目根目录
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # 国内镜像；能直连 HF 就删掉这一行
mkdir -p "$ROOT"/{models,data/hf}
pip install -U "huggingface_hub[cli]" >/dev/null

dl_model () { huggingface-cli download "$1" --local-dir "$ROOT/models/$(basename $1)"; }
dl_data  () { huggingface-cli download "$1" --repo-type dataset --local-dir "$ROOT/data/hf/$(basename $1)" ${2:+--include "$2"}; }

echo "== 1. 模型"
dl_model Qwen/Qwen3-4B-Instruct-2507          # 主模型
# dl_model Qwen/Qwen2.5-7B-Instruct           # 复现 Persona Vectors 时再下

echo "== 2. 能力评测：MMLU（只取测试集）"
dl_data cais/mmlu "all/test-*"

echo "== 3. 参考文本 + 干扰话题：UltraChat（只取一个分片就够）"
dl_data HuggingFaceH4/ultrachat_200k "data/test_sft-*"

echo "== 4. TRAIT（需要先在网页上同意条件，并设置 HF_TOKEN）"
if [ -z "$HF_TOKEN" ]; then
  echo "  跳过：没有设置 HF_TOKEN。去 https://huggingface.co/datasets/mirlab/TRAIT 同意条件后再运行"
else
  dl_data mirlab/TRAIT || dl_data snupilab/TRAIT
fi

if [ -d "$ROOT/data/hf/TRAIT" ]; then
  echo "== 5. 转换 TRAIT 为四选一格式"
  (cd "$ROOT" && python scripts/prepare_trait.py --src data/hf/TRAIT)
fi

echo "完成。目录：$ROOT"
du -sh "$ROOT"/models/* "$ROOT"/data/hf/* 2>/dev/null || true
