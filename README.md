# 鹿角记忆（Antler Memory）

在冻结的大模型激活空间里，让 AI 的人格**由经历驱动、逐渐生长**，并且随时可以拆卸。

> 模型权重从不改动。我们只在某一层的残差流上加一个向量：
> `h_ℓ ← h_ℓ + s_ℓ · Σ_k z_k · u_k`
> 其中 u_k 是大五人格的 5 个方向，z 是人格坐标，s_ℓ 是该层激活的典型大小。卸掉钩子，模型就恢复原样。

## 目录结构

```
antler/
├── antler/                  核心代码
│   ├── core/
│   │   ├── model.py         加载并冻结模型、对话模板
│   │   ├── hooks.py         Steer（注入）/ Recorder（读激活）
│   │   ├── activations.py   回答平均激活、最后 prompt token 激活、层范数 s_ℓ
│   │   └── generate.py      批量生成
│   ├── traits/
│   │   ├── data.py          大五 10 个极点、提取集 / 评测集
│   │   └── vectors.py       人格向量提取、人格坐标系 U
│   ├── eval/
│   │   ├── judge.py         LLM 评委（OpenAI 兼容接口，带本地缓存）
│   │   └── trait_score.py   生成式测量 + TRAIT 选择题测量
│   └── memory/
│       └── write.py         经历 → 人格位移 δ（方法 A 投影 / 方法 B 蒸馏）
├── experiments/             每个实验一个入口脚本
├── configs/                 main.yaml（正式）/ debug.yaml（本地调试）
├── data/
│   ├── kit/                 已整理好的数据（见 data/kit/README.md）
│   └── pilot_experiences.jsonl   E1 用的 12 段试验经历
├── scripts/
│   ├── download_on_server.sh     在服务器上下载模型和 HF 数据集
│   └── prepare_trait.py          TRAIT → 四选一格式
└── tests/                   单元测试（用随机微型模型，不需要下载）
```

## 一、本地准备（不需要 GPU）

```bash
python -m venv .venv
.venv\Scripts\activate                    # Windows；Linux/Mac 用 source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 本地只装 CPU 版
pip install -r requirements.txt
pytest -q tests                           # 应全部通过
```

`tests/test_core.py` 用一个随机初始化的微型模型验证了核心机制：
注入零向量不改变输出；注入后输出改变、退出后完全恢复（可拆卸）；记录到的激活差正好等于注入的向量；梯度能传到 δ 而不会传到模型参数。

然后用小模型把流程跑通（会下载约 1GB 的 Qwen2.5-0.5B；国内先设置 `HF_ENDPOINT=https://hf-mirror.com`）：

```bash
python experiments/e0_layer_norms.py --config configs/debug.yaml
python experiments/e0_extract.py     --config configs/debug.yaml
```

debug 配置不调用评委、只用 2 道题，**结果没有意义，只验证代码能跑**。

## 二、服务器上运行正式实验

```bash
git clone <你的仓库地址> antler && cd antler
pip install -r requirements.txt
cp .env.example .env    # 填写评委 API 和 HF token
set -a; source .env; set +a
bash scripts/download_on_server.sh
pytest -q tests
```

### E0：人格坐标系（第 1 周）

```bash
python experiments/e0_layer_norms.py                 # 各层 s_ℓ
python experiments/e0_extract.py                     # 10 个极点的人格向量 + 坐标系
python experiments/e0_calibrate.py --mode mcq        # 先用 TRAIT 便宜地粗扫层位和系数
python experiments/e0_calibrate.py --mode judge      # 再在候选层上用评委精测
```

`e0_extract.py` 结束时会打印每层“极点向量长度 / s_ℓ”。用它来调整 `configs/main.yaml` 里的 `coefs` 范围：系数应在这个量级附近扫描。

**通过标准**：5 个维度都能随系数单调变化，且连贯性不明显下降。

### E1：单段经历能否写成向量（第 2 周）

```bash
python experiments/e1_write.py --layer <E0 选出的层>                      # 方法 A
python experiments/e1_write.py --layer <层> --distill                      # 加方法 B
python experiments/e1_write.py --layer <层> --distill --behavior           # 加行为测量（花 API）
```

**通过标准**：方向准确率 > 80%；蒸馏后 KL 明显低于不注入时；注入后的特质分接近“经历放在上下文”时的特质分。

## 三、注意事项

- **数据泄漏**：`trait_extract/` 只用于提取向量和写入探针，`trait_eval/` 只用于测量，TRAIT 只用于行为评测。测试会检查前两者不重叠。
- **评委**：选定后整个项目不要更换；正式实验前抽 50 条回答人工打分，检查评委和你的判断是否一致。
- **密钥**：API key 和 HF token 只放在 `.env`，它已在 `.gitignore` 中。
- **大文件**：`models/`、`data/hf/`、`outputs/` 都不进 Git，实验结果记得另外同步回本地。

## 引用的数据来源

PERSONA（Feng et al., 2026，MIT 许可）、PersonalityBench / NPTI（Deng et al., ICLR 2025）、TRAIT（Lee et al., NAACL 2025 Findings）、Persona Vectors（Chen et al., 2025）、CLS 情景记忆（Pan et al., 2026）。
