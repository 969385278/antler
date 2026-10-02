# AGENTS.md — 给 Codex 等编码助手的指令

本项目是一个研究项目：鹿角记忆（Antler Memory）。完整规格见 `docs/RESEARCH_SPEC.md`，那里是唯一的事实来源。动手之前，先读与任务相关的章节。如果任务和规格冲突，停下来提问，不要自行决定。

## 必须遵守

1. 不要修改 `data/kit/` 下的任何文件，也不要改变 `trait_extract/`（只用于提取和写入探针）和 `trait_eval/`（只用于测量）的用途。评测集的 calib/test 切分见规格第 5.2 节。
2. 不要更换评委模型，不要修改评分提示词（`eval_prompt`、`COHERENCE_PROMPT`），也不要修改 `BASE_SYSTEM`、`MEMORY_TEMPLATE` 和 `NEUTRAL_MEMORIES`。
3. 模型权重永远冻结。所有干预都必须通过 `antler/core/hooks.py` 中的 `Steer`。
4. 层号从 0 开始编号，指第 ℓ 个 decoder block 的输出。维度顺序固定为 `DIMS = [O, C, E, A, N]`。z 和 δ 的单位是 s_ℓ。
5. 每个任务都要附单元测试。`pytest -q tests` 必须在 CPU 上通过，而且不能下载模型，使用 `tests/test_core.py` 里的随机微型模型。
6. 新的实验脚本放在 `experiments/`，通过 `_common.parse_args / setup` 读取配置，输出写到 `cfg["out_path"]` 下，并写 `run_meta.json`（规格第 15 节）。
7. 保持向后兼容：已有命令行参数的含义不能改，新功能默认关闭。
8. 注释和日志用中文；文件编码 UTF-8，换行符 LF。
9. 一个任务一个 commit，commit 信息以任务编号开头，例如 `C1: split trait_eval into calib/test`。任务编号见规格第 16.3 节。
10. 不要把 `.env`、API key、HF token、`models/`、`data/hf/`、`outputs/` 提交进 Git。

## 常用命令

```bash
pytest -q tests                                                   # 单元测试（CPU）
python experiments/e0_layer_norms.py --config configs/debug.yaml  # 本地冒烟测试（Qwen2.5-0.5B，CPU）
python experiments/e0_extract.py     --config configs/debug.yaml
```

所有脚本都必须在仓库根目录下运行。
