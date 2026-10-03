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

#### C2：保留逐题回答的精测

粗扫选出候选层之后，可以显式开启逐题保存和零注入基线复用：

```bash
python experiments/e0_calibrate.py --mode judge --config configs/main.yaml \
  --layers 15 16 17 --coefs -0.6 -0.4 -0.2 0 0.2 0.4 0.6 \
  --n-samples 1 --save-samples --reuse-baseline
```

- `--layers`、`--coefs` 覆盖本次运行的配置，不修改 YAML。层号从 0 开始。
- `--n-samples` 是每题采样次数，默认仍为 1；多次采样仅用于 judge 模式。
- `--save-samples` 在输出目录写入 `calibration_judge_samples.jsonl`，每行对应一个扫描点，包含层、维度、系数、逐题回答、特质评分和该点平均连贯性。同时写入 `run_meta.json`，记录配置、实际扫描参数、模型软件版本、评委设置、数据哈希和运行状态。
- `--reuse-baseline` 只在本次运行内复用同维度、同测量极点的零注入结果；逐题记录中的 `baseline_reused` 和 `source_layer` 标明复用关系。复用记录不能计为独立重复样本。
- 两个开关默认关闭；原有 CSV 路径和五列含义保持不变，C6 在末尾追加 `pole` 列。运行前备份同名 CSV、逐题文件和 `run_meta.json`，因为本次运行会覆盖它们。
- 示例包含 ±0.6 的边界检查点，不表示这些系数已通过连贯性验证。当前前 10 道评测题仅用于校准。

#### C6：低极测量

```bash
python experiments/e0_calibrate.py --mode judge --config configs/main.yaml \
  --layers 16 --dims C N --coefs -0.2 0 --pole low \
  --n-samples 3 --save-samples --reuse-baseline
```

- `--pole high` 是默认行为；`low` 使用低极自己的评测题和评分提示，例如 C 的 careless、N 的 calm。分数越高越体现所选极点；系数的物理方向不变。
- `--pole both` 分别生成并评分高、低极题目，产生两组结果，生成和评分量相应增加。两极使用各自的零注入基线。
- CSV 末列与样本记录中的 `pole` 为 `high` 或 `low`；样本另记 `trait_name`，元数据记录对应题库哈希。
- 推荐层按两极分别计算：高极为最大系数分减最小系数分，低极反向计算。该输出仍是扫描建议，不代表 E0 正式验收。
- MCQ 保持高特质选项概率的原有含义，`--pole low/both` 仅用于 judge 模式。本任务未改生成长度，仍按配置使用 200 tokens；长度与停止原因改动属于 C20。

#### C17：给已有提取回答补打连贯性分

```bash
python -u experiments/e0_extract_coherence.py --config configs/main.yaml
# 可选：只检查指定极点
python -u experiments/e0_extract_coherence.py --config configs/main.yaml --poles dependable careless
```

- 默认读取输出目录下 `vectors/` 中 dependable、careless、nervous、calm 的 `*_samples.json`，只调用现有评委的连贯性评分；不加载生成模型、不重新生成回答、不修改原文件。
- `extract_coherence.json` 按极点和 pos/neg 给出总数、有效数、缺失数、平均连贯性和低于 50 的比例。均值和低分比例以有效评分数为分母，50 本身不算低分；全缺失时为 `null`。
- `extract_coherence_samples.jsonl` 保存每条的 `pole`、从 0 开始的 `source_index`、`side`、原问题/回答/人格分、连贯性分及源文件 SHA-256，供 C18 按原始顺序对齐。
- 所有输入会先校验再评分，API 失败时保留已有汇总和逐条结果；再次运行利用现有 `judge_cache.jsonl` 复用已成功评分。
- 每次写 `run_meta.json`，记录输入哈希、评委和完成状态。运行前备份同名结果及旧元数据；接着运行校准前，可将本次元数据另存为 `extract_coherence_run_meta.json`。

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
