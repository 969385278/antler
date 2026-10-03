# C7：对偶基读出与E1方法A探索登记

用户于2026-10-04提出在E0诊断期间并行检查E1表示层。按规格§8.8.5，本轮仅做方法A探索，不进入正式E1或论文结论；E0目标和门槛保持。

## 固定运行

- Qwen3-4B-Instruct-2507，冻结权重，第16层，原始`outputs/main/directions.pt`；SHA256 `25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18`。
- 既有`data/pilot_experiences.jsonl`全部12段、16个预期方向标签；不按结果删经历，不补写或重标。
- 既有`probe_questions(n=30, seed=0)`，只从高极trait_extract构造探针；不用trait_eval或正式test。
- 沿用BASE_SYSTEM、MEMORY_TEMPLATE及五条NEUTRAL_MEMORIES。每道探针的零经历对照为五条中性记忆最后prompt-token激活的均值。
- 模型只做前向读取，不生成、不调用主评委、不蒸馏、不测行为。相同探针中性对照只计算一次，所有经历复用；保存其哈希与明确来源。
- 对同一个经历激活差，同时计算旧dot读出`UΔ/s`及对偶读出`solve(UUᵀ,UΔ/s)`。不为两个读出重新测量激活。

## 实现与报告

按C7的明确要求，`delta_projection`默认改为dual，新增显式dot选项保留旧点积消融；不改变既有baseline参数含义或模型提示。先验证非正交基恢复已知坐标、正交等价、子空间外残差、秩不足错误、五条中性对照及随机微型模型实际注入恢复。

独立入口`experiments/e1_projection_diagnostic.py`使用_common.parse_args/setup和唯一输出目录，避免旧e1_layer16.json相互覆盖。记录run_meta、全部输入和产物哈希、原始平均激活差、两个读出的逐经历坐标、原始命中数和各维标签构成。

共模和去共模符号结果仅作描述；12段经历方向不平衡，尤其C标签全为正，不能将去均值变化视为更好的准确率，也不能仅凭共模/离散度比值宣告格式效应消失。16个标签不是16个独立经历，不据此宣布超过80%的总体准确率。

这是原始双极坐标的探索，不能直接验证尚未定稿的第21层单极/decode方案。对偶基只修正非正交坐标读出，不证明人格方向干净，也不消除行为串扰。本轮不报告KL人格解释率或优于随机子空间；这些仍需C11的共同评估集合、对照及正式E1设计。

固定12段完成后停止；缺失激活、秩不足、非有限数值、输入变化时保留失败并停止。不追加层、探针数、蒸馏或行为测试，也不覆盖旧产物。完成CPU离线测试后才部署运行。

## 验证与运行命令

C7核心26项及探索入口41项，CPU离线全套`pytest -q tests`为673 passed（63.26秒）；无下载模型。覆盖已知坐标、微型模型冻结、17组前向复用、全部510提示保存、输入变更即停及失败产物。模型记录路径、配置和tokenizer文件哈希、形状、版本与冻结状态，本轮未做8GB权重文件全量SHA。

在服务器仓库根目录执行一次：

```bash
python -u experiments/e1_projection_diagnostic.py \
  --config configs/main.yaml \
  --directions outputs/main/directions.pt \
  --experiences data/pilot_experiences.jsonl
```
