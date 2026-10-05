# 鹿角记忆（Antler Memory）

> 研究进度（2026-10-05）：正在验证第21层decode的经历状态小样本，尚未正式通过E1。双胞胎上下文与第16层诊断已有结果，见 [当前进度](docs/PROGRESS.md) 和 [最新风险核查](docs/E1_L21_UPSTREAM_FINDINGS_2026-10-05.md)。


**研究进度（2026-10-04）：[当前进展与下一步](docs/PROGRESS.md)**。E0 本轮方法搜索已收尾，Strict / Practical 均未正式通过；E1 已有探索信号，正式验证尚未完成。

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

#### 一次跑完当前 C17 + C6 诊断（推荐）

服务器在仓库根目录执行：

```bash
git pull --ff-only && bash scripts/run_e0_diagnostics.sh
```

- 自动加载根目录 `.env`；优先使用 `.venv/bin/python`，否则用当前环境的 `python`。
- 先给 dependable、careless、nervous、calm 的已有提取回答补分，再测第 16 层 C/N 的低极，系数 `-0.2 / 0`，前 10 道 calib 题每题采样 3 次，共生成 120 条回答。沿用现有评委和配置中的生成长度。
- 启动前检查两步输入；旧结果自动备份到 `outputs/main/diagnostics/<本轮编号>/backup/`。不要同时手动运行其他写入 `outputs/main` 的实验脚本。
- 每步保存日志、结果和各自的 `run_meta.json`；失败立即停止。再次运行同一命令，会校验输入及归档文件哈希，跳过已完成步骤；失败步骤重新执行，评委缓存继续复用。
- 配置、输入、代码或评委设置变化后自动新建一轮。需要主动重做时加 `--fresh`；一键入口没有逐测量点恢复，C6 中断后会重跑其 4 个测量点。
- 结束后显示 `summary.md` 和 `results.zip` 的完整路径；把 `results.zip` 发回来即可。结果包包含逐条回答、评分、每步元数据和日志；备份留在服务器，不打包。最近一轮位置记录在 `outputs/main/diagnostics/latest.json`。
- 本入口仅编排 C17 与 C6；已有这轮结果后，可运行下方 C18/C20 对照。它沿用当前校准流程，不代表 E0 正式通过。

#### C20：区分人格变化、任务理解和生成截断

在 judge 校准命令中加 `--with-task --max-new-tokens 320 --save-samples`：

- 每条回答保存人格分、连贯性、生成 token 数及停止原因 `eos/length`；启用 `--with-task` 后另存任务理解分。
- 三类评分分别计算有效数和均值，缺失分不当作零分。CSV 仅在开启任务评分时追加 `task` 列。
- 默认关闭任务评分；未指定长度时仍沿用原配置。提取阶段的 200 token 配置保持原样。
- 任务理解提示采用规格第 4.6 节原文。本轮分数标记为探索性、未经人工验证；首次正式使用前需作者盲标 30 条，Spearman ≥ 0.6。

#### C18：旧回答重算向量与一键对照

已有 C17 的逐条补分结果和十极点向量后，服务器执行：

```bash
git pull --ff-only && bash scripts/run_e0_refilter_compare.sh
```

一次执行三组对照：原始方向、仅配对人格筛选、配对后再加连贯性筛选。
重算只对旧回答做前向计算，不重新生成提取回答，不调用评委。仅重算 C/N 的四个极点，其余六极向量和层尺度保持原值。

- 两组重算均要求同题正、负回答成对满足 `pos >= 50`、`neg < 50`；连贯性组还要求两侧 `coherence >= 50`。旧提取默认独立筛选且正侧为 `> 50`，所以原始组与配对组有筛选规则差异；配对组与连贯性组用于比较新增质量门槛。
- 三组都在第 16 层测：C 高极 `0/+0.2`、低极 `-0.2/0`；N 高极 `0/+0.2/+0.25`、低极 `-0.2/0`。每点前 10 道 calib 题各采样 3 次，共 **810 条新回答**，最多 2430 个评分请求，缓存可减少实际请求。
- 本轮统一启用任务理解分和 **320 token** 评测上限。原 200 token 结果不混入对照；人格效应、任务理解、连贯性、截断比例一起看。同系数不代表人格强度相同。
- 输出在 `outputs/main/refilter_diagnostics/<本轮编号>/`，原结果和主方向文件保留。不要同时运行其他会改动来源向量、C17 分数、配置或代码的任务。
- 先校验来源、C17 逐条哈希、样本顺序和模型文件；每极点至少保留 5 对。旧回答按当前提取题库还原系统提示并核对顺序，元数据明确记录历史提示哈希缺失的限制。
- 14 个子步骤分别保存结果、元数据和日志。中断后重复同一命令，已完成且哈希一致的步骤跳过；单步中断会重跑该步，缓存复用。父向量重算后相应测量也会重跑。输入改变自动新建一轮，主动重做加 `--fresh`。
- 完成后终端显示 `summary.md`、`results.zip` 的路径；把结果包发回来。包中有 30 条任务理解盲标表 `task_blind.csv`、评分说明及独立映射。任务评分仍为探索性，本入口不会自动宣布 E0 通过或替换主坐标系。

单独重算可用（目标目录必须为空）：

```bash
python experiments/e0_refilter.py --config configs/main.yaml
# 默认写入 outputs/main_cohfilt；仅配对的匹配对照：
python experiments/e0_refilter.py --config configs/main.yaml --paired-only
```

`extract_pole(..., min_coherence=None)` 保持原行为；显式传入连贯性下限才开启配对筛选。

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
