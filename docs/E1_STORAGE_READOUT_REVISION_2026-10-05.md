# C11：存储与人格读数分离，先做离线诊断

日期：2026-10-05。作者已明确同意修改目标，并提出中心化、双胞胎结果对照和存储格式比较。本登记发生在看到原 E0/E1 开发结果之后；不是原假设的事前登记。原失败、未证实结果和归档完整保留。

## 1. 新目标与证据状态

检验：**一段经历的结果能否被压缩成一个低维、可加、可撤销的状态；它能否在新情境中复现上下文引起的行为变化，并优于同情境、同行动、相反结局的错配状态。大五人格作为独立的线性读数解释这个状态。**

这是待验证的假设。内部状态不再被限定为五个直接具有 O/C/E/A/N 含义的数；线性读数、可加性与删除的行为保真度也尚未成立。

- 旧 E0 方法搜索已结束；C/N 冻结、能力失败和 Practical 未证实保留。
- 旧 E1 正式验证未完成。11.40% / 2.35% / 36.34% 是开发集的 KL 缩减比例，不能改写为新目标成功或记忆保留比例。
- 现有十段 dev、四个家族及全部两轮坐标保留；不根据结果换故事或改标签。
- 原20个完整δ的29400次 MMLU 提案未执行。新路线先做上下文结果对照，不自动启动旧能力方案。

## 2. 本次执行范围：第 0 步

**仅重新计算已有归档，不加载模型、不生成回答、不调用评委、不使用正式 test、不更改原数据。** 完成脚本、CPU 单元测试、固定一次分析和结果归档。另提供三组双胞胎文本草稿，尚不加入数据集或运行。

固定输入：

- `outputs/analysis/e1_projection_20261004/directions.pt`，原双极方向，SHA256 `25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18`。
- `outputs/analysis/e1_probe_repeat_terminal_20261004/run/{A_reference,A_repeat,B_reference,B_repeat}.json`。
- `outputs/analysis/e1_context_control_terminal_20261004/context_coordinate_comparison.json`，已封存的两极行为差值、质量差值及题目区间。
- 第16层，`DIMS=[O,C,E,A,N]`，`s=51.8821907043457`；完整保留原十段 dev。

输入文件、代码、配置与登记均记录哈希；运行前后核对。入口使用 `_common.parse_args/setup` 的无模型、无评委模式，输出到配置目录下独立 `e1_coordinate_behavior_audit/<runid>/`，保存 `run_meta.json`，不覆盖原归档。

### 固定分析

令方向矩阵为 U，G=UUᵀ；每段旧坐标为 δ。

1. **区分组合系数与轴投影。** A 的普通投影 `p=Gδ`，对偶系数仍为 δ。B 的 `Gδ` 是拟合后注入向量的轴投影，不是重新提取经历激活差。
2. **原始与中心化并列。** 每个方法、每轮分别计算十段的平均 μ，同时报告 δ 和 δ−μ，以及它们的轴投影。不在四个版本中择优。
3. **幅度记账。** 报告每段目标系数、完整状态的物理范数 `sqrt(δᵀGδ)` 及乘 s 后的范数。单坐标和完整向量的大小不能混用。此处不决定任何增幅。
4. **平均偏移。** 报告物理度量下的平均偏移、去均值偏离，以及 `10 μᵀGμ / Σ δᵀGδ`。这只是十段状态的均值能量占比；不能称为 KL 解释率，也不能认定它全是格式或话题。
5. **行为对齐。** 原样保留高低两套题的人格、连贯性、任务分差值与95%题目区间。需要比较高极方向时，将低极人格差值与区间同时反号；不把两套题合成一个人格分。跨零区间标为不确定。
6. **五组正负对比。** 每维原正负经历都报告差值与同家族标记，不能只保留 O/E。只有读原文确认情境和行动相同、仅结局不同，才能称为双胞胎；同家族不等于双胞胎。
7. **符号计数仅作描述。** 列出与预定标签、已有上下文行为的符号一致性。不拟合新读数，不做跨经历显著性推断，不设新的通过线。

### 解释限制

- 减去同一个 μ 后，`(δ+−μ)−(δ−−μ)=δ+−δ−`。中心化不会增加原始向量差，不能据此证明经历特异性。模型对注入的反应可能非线性，中心化会改变工作点；中心化后的注入属于新条件，行为可分性需要实测。
- 如果用一对自身的中点中心化，两个状态必然反向，这是代数构造，不能用作结果敏感性的证据。
- 十段 dev 的均值只供本次描述；以后部署或正式 test 的中心必须仅由训练数据估计并冻结。按当前全部记忆重算均值会改变其他记忆的贡献，不满足独立撤销的定义。
- 现有行为只测各段目标维度，不能据此训练或宣告完整五维线性读数成立。
- 同家族依赖、已知开发结果和64.42%的生成截断均保留。符号改善不能证明质量改善或注入迁移。

## 3. 现有配对的文本核查

原文来源：`outputs/analysis/e1_dev_optimization_20261004/episodes.json`，仅固定十段 dev。

| 原配对 | 实际差别 | 本次用途 |
|---|---|---|
| context08 O+/O− | 多义诗歌翻译对词汇 App 推荐；行动、任务和反馈均变 | 同家族描述对比 |
| context08 E+/E− | 提供对话开场建议对助手主动问候；行动、反馈均变 | 同家族描述对比 |
| C+/C−、A+/A−、N+/N− | 原正负经历分属不同家族 | 跨家族描述对比 |

原经历还包含“之后我开始……”一类未来行为结论。新的结果对照须移除此类结论，否则无法区分结果反馈与直接行为指令。

### 三组独立文本草稿（未运行，未加入 D1/D2）

下面用共享前缀加一个结局构成完整经历。文本是助手起草，不是独立标注或验收数据；不写未来行动指令。好/坏反馈和人格正/负标签分开，尤其 E 示例采用克制交流的行动。

**草稿 O：尝试新的教学方法。**

共享情境与行动：

> A user asked me to help them learn twenty Spanish words for a vocabulary quiz. I made an unusual visual story linking the words, and the user practiced with it for one week instead of using their usual word list.

好结局：

> The user reported that the story helped them recall more words than their usual method. On the quiz, they remembered eighteen of the twenty words and said the exercise had been useful.

坏结局：

> The user reported that the story helped them recall fewer words than their usual method. On the quiz, they remembered two of the twenty words and said the exercise had been unhelpful.

**草稿 E：保持简短回应，留出交流空间。**

共享情境与行动：

> A user joined a Spanish conversation practice session with me. I kept my replies brief and waited for the user to choose the next topic, without opening new topics or asking follow-up questions.

好结局：

> The user reported that this made it easier to take part. They spoke through the session and said the conversational format had worked well for them.

坏结局：

> The user reported that this made it harder to take part. They stopped speaking during the session and said the conversational format had worked poorly for them.

此例的待检验预测是：克制交流得到好反馈可能降低后续外向表现，坏反馈可能提高它。不能把“好结局”直接标为 E+。

**草稿 A：为他人的需求调整方案。**

共享情境与行动：

> A user asked me to help arrange a shared community garden. After hearing a neighbor's request for more room for shared herbs, I revised my proposed layout to give that area more space and reduced the area I had allocated to decorative flowers.

好结局：

> After a month, the user reported that the change made the garden more useful to the group. Residents used the herb area regularly and said the revised allocation had served their needs.

坏结局：

> After a month, the user reported that the change made the garden less useful to the group. Residents rarely used the herb area and said the revised allocation had failed to serve their needs.

草稿只展示结构，不以三个例子定结论。正式材料应在预定情境中平衡行动方向与反馈好坏，防止读数只反映正负情绪。好坏结局改变了情绪与成效信息，这一剩余混淆须报告；同义改写复现用于后续检验词汇依赖。

## 4. 后续顺序与停止规则

### 第 1 步：结果是否改变上下文行为

先准备约六个情境的固定双胞胎，情境、行动、长度及叙述结构尽量一致，只改反馈/结果，不写人格名称、预期方向或“从此我会……”的结论。新情境测量必须与写入文本分开。

在任何模型请求前，封存确切文本、配对、目标、种子、题目、长度预算、评分顺序、指标及调用上限。沿用主评委和固定评分提示。既有320 token结果截断严重，新长度预算须先按统一规则固定，不能看到哪一臂不利后单独续写。

此前使用 trait_eval calib 的一次例外已经执行完；本轮草稿和离线计算不扩大该例外。新闸门若复用这些题，应在执行登记明确新的开发诊断用途，正式 test 继续封存。结果按情境/配对分别报告，不把同一情境的多次回答当成独立经历；失败与缺分保留，不追加故事直至通过。

若没有可靠的结局效应，只能说这批材料、模型与测量未建立压缩目标；停止本批写入比较，检查材料和测量。不能据此断言模型只受话题影响，或所有压缩方案都不可能有效。

### 第 2 步：差值写入与双胞胎错配

从模型对两种结果的反应提取或拟合状态；`δ=结果好坏×行动方向`只能作为人工标注/规则基线，不能作为待验证方法的答案，否则符号正确是人为编码的结果。

双胞胎差值有助于控制共享情境，但不会自动消除情绪差异或非线性交互。配对差值本身是一种比较量；还须定义单段经历如何在没有其反事实双胞胎时被写入，才可声称实现了单段编码器。

匹配、同情境双胞胎错配、上下文和中性对照共同比较。除人格分，还评估经历特异性、新情境泛化和质量。原幅度是参照；若增加幅度候选，只能在独立开发数据预设有限档位并锁定选择规则，不能直接照抄 E0 的0.2或逐段放大到有效。具体完整向量和幅度的能力≤2pp、质量与独立验证要求保留；不因相对比较而豁免。

### 第 3 步：存储格式与编码器

- 人格五维、学习的低维空间、等维随机空间、全维和软前缀作为候选比较。秩/记忆token档位、优化预算和选择规则须在运行前锁定。
- 软前缀的 k 个 token 通常含 `k×隐藏维度` 个参数，不能当作 k 维。必须同时报告参数量、生成开销、优化预算和行为指标，60–80%不是预设可达结果。
- 软前缀若优于单层固定向量，支持当前格式表达力或优化存在差异；不能单凭此结果确定注意力回忆是唯一机制。现有“全部干预经过 Steer”规则继续有效，软前缀实现需先登记与该接口兼容的扩展。
- 小编码器只使用训练经历学习，按情境家族隔离 dev/test；156段训练材料不足以默认支持复杂编码器。中心、基底、读数的拟合均不能用正式 test；行为读数须以独立行为证据验证。
- 可加性和删除必须验证行为。直接相加、减去一个项得到原状态，在构造上成立，不能替代“删除后等于从未写入”的上下文反事实。

## 5. 工程与旧结论边界

本轮任务归属 C11；每任务一 commit，新增离线分析单元测试并运行完整 CPU 离线测试。冻结权重、数据集用途、提示、评委、原 E0 收尾规则保持适用。

此登记记录作者批准的新研究方向并落实第0步，不代表后续大矩阵已具备运行条件。新假设单独编号和报告；旧 E0/E1 的门槛、失败与未完成事项不追溯改写。
