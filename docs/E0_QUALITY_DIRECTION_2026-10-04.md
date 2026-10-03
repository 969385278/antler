# C18：第21层连贯性相关方向的有限探索

## 问题与边界

用户提出从人格方向里剥离回答质量分量。固定200条盲审已完成：量尺分歧存在，但C−0.225的助手低极连贯损失仍为11.1分，原任务理解失败保留。下一步先检验能否从既有提取回答中读出稳定的连贯性相关方向，再决定是否值得做剥离干预。

本轮只重放已有回答、读取激活。不生成、不注入、不新增评分，不使用trait_eval/calib/test。不改模型、人格坐标系、固定提示或正式门槛。C17没有任务理解分，结果只能称“主评委连贯性相关方向”，不能称纯质量或通用能力方向。

此登记发生在计算该方向或查看其激活结果之前，已经查看既有评分分布及来源，不宣称对既有标签完全盲。

## 固定来源

- Qwen3-4B-Instruct-2507，冻结权重，第21层，batch16，seed0。
- `outputs/main/extract_coherence_samples.jsonl`，SHA256 `6add4da12b7a09f24148c43911a98fa0751f3034e5e8d564076d6fa2f8c09162`。
- 固定dependable、careless、nervous、calm四极点，各400条：5对提示×20题×2次生成×pos/neg，共1600条。顺序依次为上述极点、原source_index；不按评分筛样本。
- C17记录的源回答SHA必须与`outputs/main/vectors/<pole>_samples.json`原字节一致；逐条核对side、question、answer、trait_score和source_index。恢复提示采用`extraction_rows(n_rollouts=2)`，只读trait_extract。
- 层尺度取原`outputs/main/directions.pt`，SHA256 `25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18`。
- 比较向量取完整paired集：`outputs/main/refilter_diagnostics/20261003T022259Z_099e5cd2/variants/paired/vectors/`。

| 极点 | paired向量SHA256 |
|---|---|
| dependable | `39676e4ae642b96cf163899c133cbc9ec27a18b77b7d5c1d8818fa90f28a9cdf` |
| careless | `cc6c27f273e0e78d3ed069ab51ea360f00ea1e76d8ec0bfcf749d003c4e6b539` |
| nervous | `4caa81abb73a3ee01b6bde595dab19a7f879057d7190d7325468d955b4f9f507` |
| calm | `93479c98d977519eae79822e7bacf9fa9eaa251b4d94fbc49abfb2ee18fdfcfa` |

保存实际模板化prompt、原answer及重分词后参与`response_mean_acts`均值的token数，即`len(tokenize(prompt+answer))−len(tokenize(prompt))`。旧样本没有历史token ID和stop原因，不能声称精确重放历史token流，不能把本次重分词长度当历史生成长度。

## 唯一估计方法

同一`pole/prompt_pair/question/side`中，用rollout1减rollout0，共800个差分，80个不同问题。全部差分保留，包括连贯分差为0的组。

```text
A = (回答平均激活1 − 回答平均激活0) / s_21
y = 连贯分1 − 连贯分0
t = 人格分1 − 人格分0
l = log1p(重放回答token数1) − log1p(重放回答token数0)
```

控制矩阵X固定13列：8个pole/side截距、4个pole各自的人格分差斜率、1个长度差斜率。相同条件内作差控制固定题目和提示，X控制剩余可测人格及长度差异。用float64最小二乘分别拟合`y~X`和`A~X`，得到残差r和R：

```text
b = Rᵀr / (rᵀr)
q = b / ||b||
```

正方向按训练集较高连贯分定义，禁止根据人格余弦翻转。矩阵秩不足、残差或方向范数为零、非有限数值时保留失败并停止，不改协变量或临时加正则。该估计控制的是已有评分，不保证控制真实人格全部语义。

## 跨题验证及几何检查

1. 每极点20题按`SHA256("quality-direction-v1\n" + pole + "\n" + question)`的UTF-8字节排序，交替分A/B，各10题。同题全部5提示×2side归同折。加载模型前先写折分身份及哈希。
2. A训练/B评估，B训练/A评估。评估折只能使用训练折拟合的X系数残差化；在训练折拟合无截距标量`r~R@q`统一预测尺度。不得在评估折重拟合控制或校准系数。
3. 报告两折、合并及8个pole/side的未见题残差连贯性与校准预测Spearman，不将训练内分数当泛化证据。
4. 固定1000次、seed0、按折×pole分层的整题bootstrap，每题10对一起重抽，每次重新拟合两个训练折和标量，报告合并未见题相关的95%区间。折归属不变，重抽的题不会泄漏到相反训练折。
5. 报告两互不重叠训练半部的`cos(q_A,q_B)`，再以全800对拟合q。
6. 从本次激活按既有paired人格规则（pos≥50且neg<50）重建四极点全量及半题向量，不加连贯过滤。全量与保存向量逐一比较，余弦需≥0.995且相对范数误差≤0.05；否则保留产物停止解释几何，先查来源或数值差异。容差允许BF16和分批带来的微小数值差异，不声称完全相同。
7. 全量q只与实际`normalize(v_careless)`和`normalize(v_nervous)`判定候选；同时报告`cos(q_A,v_B)`、`cos(q_B,v_A)`，减少同池噪声造成的虚假重合。C负系数实际沿careless正向注入，不能用`−u_C`替代。

## 固定停止规则

以下是本次有限方法探索的投入标准，不是E0门槛或正式统计发现标准，数字不随结果变动。

- 身份、来源、重放向量或拟合校验失败：停止并报告。
- 合并未见题相关95%区间下限≤0，或任一折相关≤0：不进入剥离干预。
- 两半q余弦<0.30：不进入剥离干预，报告方向不稳定。
- 仅当上述检查通过，某目标全量`cos(q,v_target)≤−0.30`，且两个交叉半题余弦均为负，才记为值得进一步干预验证的方向。只满足一个目标就只保留一个，不用绝对余弦把正号解释成损伤。
- 固定1600条及统计完成即结束；不换折分、不补层或评分，不切换另一估计器挽救结果。

若通过，也只登记下一轮有限干预，不自动宣告剥离有效。本轮不执行`u−proj_q u`注入。未来须明确归一化及注入范数，因为`steer_vector_poles`会自动单位化方向；质量改善须同时报告人格效果、两套题的连贯与任务分，并保留能力和test要求。

## 产物与验证

入口`experiments/e0_quality_direction.py`使用_common.parse_args/setup，写入cfg[out_path]下唯一目录，记录run_meta、固定源/源码/模型配置和tokenizer哈希、完整身份、提示、1600条激活、方向、全部统计及失败状态。模型权重不做本轮全量SHA，明确该身份核验限制。运行前通过CPU离线单元测试；不在运行中修改源码或输入。

```bash
python -u experiments/e0_quality_direction.py \
  --config configs/main.yaml \
  --source-out outputs/main \
  --pole-vectors outputs/main/refilter_diagnostics/20261003T022259Z_099e5cd2/variants/paired/vectors
```

## 实现检查（运行前）

新增统计模块27项、入口39项定向测试通过；CPU离线全套`pytest -q tests`为739 passed（82.24秒），无下载模型。覆盖已知方向恢复、人格/长度混杂、未见题标签不参与训练、整题分折及加权bootstrap与真实重复题等价、源身份与哈希、随机微型模型回答token切片、无生成/评委调用、输入漂移及统计失败留档。独立静态审查未发现公式、题目泄漏或单极符号阻断。bootstrap出现无效拟合即终止且保留已完成统计，不丢弃失败后补抽。
