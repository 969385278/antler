#!/usr/bin/env bash
# 从已有 C17 结果启动三组对照，保留原始向量和结果。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
if [[ -x .venv/bin/python ]]; then
    python_bin=.venv/bin/python
else
    python_bin=python
fi
exec "$python_bin" -u experiments/e0_refilter_compare.py "$@"
