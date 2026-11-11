# 文档导航

本目录保存鹿角记忆（Antler Memory）研究的规格书、实验登记和逐项结果报告，是根目录研究报告的完整证据。项目约定：每次实验**在运行之前**先写登记文件，写明方法、预算、通过标准和停止规则；运行结束后另写结果报告。因此大多数实验都成对出现：一份"登记"、一份"结果"。

文件名中的日期是写作日期；`C` 加数字是研究规格书第 16.3 节中的任务编号。为了保证已归档实验的来源哈希可以复核，文件保持原位置和原文件名，不做移动或改名。

## 先读这几份 ⭐

| 文档 | 内容 |
|---|---|
| [研究报告](../README.md) | **报告正文**：摘要、方法、实验设置、结果、讨论与结论 |
| [RESEARCH_SPEC.md](RESEARCH_SPEC.md) | **研究规格书**：研究问题、可证伪假设、相关工作、方法形式化、数据切分规则、全部实验设计与变更记录 |
| [E1_PROGRESS_AND_ISSUES.md](E1_PROGRESS_AND_ISSUES.md) | E1 各项实验的汇总与问题分析 |
| [E0_ROUND_CLOSEOUT_2026-10-04.md](E0_ROUND_CLOSEOUT_2026-10-04.md) | E0（人格坐标系）收尾：注入效果、质量代价与 Prompt 对照 |
| [E1_TWIN_CONTEXT_GATE_RESULTS_2026-10-05.md](E1_TWIN_CONTEXT_GATE_RESULTS_2026-10-05.md) | 六情境"行动 × 结局"双胞胎经历的上下文实验 |
| [E1_TWIN_STATE_EXTRACTION_RESULTS_2026-10-05.md](E1_TWIN_STATE_EXTRACTION_RESULTS_2026-10-05.md) | 内部状态几何：经历差值主要落在共同的"好/坏反馈"方向上 |
| [E1_L21_SMALL_PILOT_RESULTS_2026-10-05.md](E1_L21_SMALL_PILOT_RESULTS_2026-10-05.md) | 第 21 层原幅度写入的小样本行为结果 |
| [E1_METHOD_LITERATURE_2026-10-05.md](E1_METHOD_LITERATURE_2026-10-05.md) | 面对 E1 问题的文献依据与候选改进路线 |
| [USAGE.md](USAGE.md) | 环境配置与运行命令 |

## E0：人格坐标系

目标：在 Qwen3-4B 的残差流中找到大五人格的线性方向，检验单层注入能否单调、可控地调节人格，以及代价有多大。

| 阶段 | 登记（运行前） | 结果 |
|---|---|---|
| 验收定义 | [E0_GATE_REVIEW](E0_GATE_REVIEW_2026-10-03.md) | — |
| 交接与问题梳理 | [E0_HANDOFF](E0_HANDOFF_2026-10-03.md) | — |
| C5 生成式能力诊断 | [E0_GENERATED_CAPABILITY](E0_GENERATED_CAPABILITY_2026-10-04.md) | [结果](E0_GENERATED_CAPABILITY_RESULTS_2026-10-04.md) |
| C5 延迟选项预测 | [E0_DELAYED_CAPABILITY](E0_DELAYED_CAPABILITY_2026-10-04.md) | [结果](E0_DELAYED_CAPABILITY_RESULTS_2026-10-04.md) |
| C9 配对盲审 | [E0_MATCHED_REVIEW](E0_MATCHED_REVIEW_2026-10-04.md) | [结果](E0_MATCHED_REVIEW_RESULTS_2026-10-04.md) |
| C18 连贯性相关方向 | [E0_QUALITY_DIRECTION](E0_QUALITY_DIRECTION_2026-10-04.md) | [结果](E0_QUALITY_DIRECTION_RESULTS_2026-10-04.md) |
| C18 连贯分量剥离 | [E0_QUALITY_PROJECTION](E0_QUALITY_PROJECTION_2026-10-04.md) | [结果](E0_QUALITY_PROJECTION_RESULTS_2026-10-04.md) |
| C19 注入时段对照 | — | [E0_PHASE_DIAGNOSTIC](E0_PHASE_DIAGNOSTIC_2026-10-04.md) |
| C19 decode 强度对照 | [E0_DECODE_STRENGTH](E0_DECODE_STRENGTH_2026-10-04.md) | [结果](E0_DECODE_STRENGTH_RESULTS_2026-10-04.md) |
| C20 任务理解锚点 | [E0_TASK_ANCHORS](E0_TASK_ANCHORS_2026-10-04.md) | [结果](E0_TASK_ANCHOR_RESULTS_2026-10-04.md) |
| C20 任务分原文核读 | — | [E0_C_TASK_READING](E0_C_TASK_READING_2026-10-04.md) |
| C21 Prompt Persona 对照 | [E0_PROMPT_PERSONA](E0_PROMPT_PERSONA_2026-10-04.md) | 见收尾报告 |
| D5 盲评复核 | [D5_BLIND_WORKFLOW](D5_BLIND_WORKFLOW.md) | [E0_D5_ASSISTANT_REVIEW](E0_D5_ASSISTANT_REVIEW_2026-10-04.md) |
| 本轮冻结与收尾 | [E0_ROUND_FREEZE](E0_ROUND_FREEZE_2026-10-04.md) | [E0_ROUND_CLOSEOUT](E0_ROUND_CLOSEOUT_2026-10-04.md) ⭐ |

## E1：单段经历能否写成向量

目标：把一段经历在上下文中造成的影响，压缩成残差流中的一个低维、可加、可撤销的状态。

**写入方法与 KL 解释率**

| 阶段 | 登记（运行前） | 结果 |
|---|---|---|
| C7 方法 A（对偶基投影） | [E1_PROJECTION_PILOT](E1_PROJECTION_PILOT_2026-10-04.md) | [结果](E1_PROJECTION_RESULTS_2026-10-04.md) |
| C11 KL pilot（12 段） | [E1_KL_PILOT](E1_KL_PILOT_2026-10-04.md) | [结果](E1_KL_PILOT_RESULTS_2026-10-04.md)、[完整汇总](E1_KL_ANALYSIS_2026-10-04.md) |
| 留出 KL 评估 | [E1_KL_VALIDATION](E1_KL_VALIDATION_2026-10-04.md) | — |
| 全维优化诊断 | [E1_OPTIMIZATION_DIAGNOSTICS](E1_OPTIMIZATION_DIAGNOSTICS_2026-10-04.md) | [E1_ADAM_GEOMETRY](E1_ADAM_GEOMETRY_2026-10-04.md) |
| 固定 10 段 dev 优化（90 条件） | [E1_DEV_OPTIMIZATION](E1_DEV_OPTIMIZATION_2026-10-04.md) | 见进展总览 |
| 随机子空间对照 | [E1_SUBSPACE_PILOT](E1_SUBSPACE_PILOT_2026-10-04.md) | [结果](E1_SUBSPACE_RESULTS_2026-10-04.md) |
| 探针复测 | [E1_PROBE_REPEAT](E1_PROBE_REPEAT_2026-10-04.md) | [结果](E1_PROBE_REPEAT_RESULTS_2026-10-04.md)、[分析组件](E1_STABILITY_ANALYSIS_2026-10-04.md) |

**行为与上下文对照**

| 阶段 | 登记（运行前） | 结果 |
|---|---|---|
| 五条中性记忆配对 | — | [E1_BEHAVIOR_SUMMARY](E1_BEHAVIOR_SUMMARY_2026-10-04.md) |
| 无注入上下文诊断（2400 回答） | [E1_CONTEXT_CONTROL_PROPOSAL](E1_CONTEXT_CONTROL_PROPOSAL_2026-10-04.md) | [结果](E1_CONTEXT_CONTROL_RESULTS_2026-10-04.md) |
| 完整 δ 能力检查（未运行的提案） | [E1_FIXED_DELTA_CAPABILITY_PROPOSAL](E1_FIXED_DELTA_CAPABILITY_PROPOSAL_2026-10-04.md) | — |
| E1 完成条件 | [E1_READINESS](E1_READINESS_2026-10-04.md) | — |

**目标修订后：存储与人格读数分离（双胞胎经历）**

| 阶段 | 登记（运行前） | 结果 |
|---|---|---|
| 目标修订与离线第 0 步 | [E1_STORAGE_READOUT_REVISION](E1_STORAGE_READOUT_REVISION_2026-10-05.md) | [E1_COORDINATE_BEHAVIOR_AUDIT](E1_COORDINATE_BEHAVIOR_AUDIT_2026-10-05.md) |
| 六情境"行动 × 结局"上下文闸门 | [E1_TWIN_CONTEXT_GATE](E1_TWIN_CONTEXT_GATE_2026-10-05.md) | [结果](E1_TWIN_CONTEXT_GATE_RESULTS_2026-10-05.md) ⭐、[独立审计](E1_TWIN_GATE_AUDIT_2026-10-05.md) |
| 内部状态读取 | [E1_TWIN_STATE_EXTRACTION](E1_TWIN_STATE_EXTRACTION_2026-10-05.md) | [结果](E1_TWIN_STATE_EXTRACTION_RESULTS_2026-10-05.md) ⭐ |
| 原幅度写入与结局错配 | [E1_TWIN_RAW_BEHAVIOR](E1_TWIN_RAW_BEHAVIOR_2026-10-05.md) | [能力检查结果](E1_TWIN_RAW_CAPABILITY_RESULTS_2026-10-05.md) |
| 第 21 层风险小样本 | [E1_L21_SMALL_PILOT](E1_L21_SMALL_PILOT_2026-10-05.md)、[上游核查](E1_L21_UPSTREAM_FINDINGS_2026-10-05.md) | [结果](E1_L21_SMALL_PILOT_RESULTS_2026-10-05.md) ⭐ |
| 文献依据与候选路线 | — | [E1_METHOD_LITERATURE](E1_METHOD_LITERATURE_2026-10-05.md) |

## D1：经历数据集构建

目标：构建 280 段带人格方向标签的"AI 经历"短文，按情境家族划分 train / dev / test，避免泄漏。生成后由两个不同的大模型独立盲标，再经过结构审核、版本选择和有界补写。

| 阶段 | 文档 |
|---|---|
| 数据流程总览 | [D1_DATA_PIPELINE](D1_DATA_PIPELINE_2026-10-04.md) |
| 首轮生成 | [登记](D1_GENERATION_2026-10-04.md) · [结果](D1_FIRST_ROUND_RESULTS_2026-10-04.md) · [语义审核](D1_SEMANTIC_AUDIT_2026-10-04.md) |
| 双模型独立盲标 | [标注接口](D1_ANNOTATION_2026-10-04.md) · [续跑](D1_ANNOTATION_RESUME_2026-10-04.md) · [跨批汇总](D1_BATCH_CONSOLIDATION_2026-10-04.md) · [两批结果](D1_TWO_BATCH_RESULTS_2026-10-04.md) · [三批结果](D1_THREE_BATCH_RESULTS_2026-10-04.md) |
| 结构审核与标签口径 | [数字角色批次审核](D1_DIGITAL_BATCH_SEMANTIC_AUDIT_2026-10-04.md) · [结构疑点复核](D1_STRUCTURAL_ADJUDICATION_2026-10-04.md) · [目标标签政策](D1_TARGET_LABEL_POLICY_2026-10-04.md) · [重合并结果](D1_TARGET_LABEL_RESULTS_2026-10-04.md) |
| 补写与版本选择 | [机械失败补写](D1_RECOVERY_2026-10-04.md) · [补写审核](D1_RECOVERY_SEMANTIC_AUDIT_2026-10-04.md) · [98 槽位补写](D1_REPLACEMENT_2026-10-04.md) · [结果](D1_REPLACEMENT_RESULTS_2026-10-04.md) · [审核](D1_REPLACEMENT_REVIEW_2026-10-04.md) · [版本选择](D1_VERSION_SELECTION_2026-10-04.md) |
| 剩余缺口 | [提案](D1_REMAINING_PROPOSAL_2026-10-04.md) · [槽位登记](D1_REMAINING_SLOTS_2026-10-04.md) · [收尾](D1_REMAINING_RESULTS_2026-10-04.md) · [方向复核](D1_DIRECTION_REVIEW_2026-10-04.md) · [第三轮](D1_ROUND3_2026-10-04.md) · [第三轮结果](D1_ROUND3_RESULTS_2026-10-04.md) |

## 其他

- [blogs/](blogs/)：面向一般读者的研究博客草稿
- [PROGRESS.md](PROGRESS.md)：按时间记录的研究进度（较早版本，最新情况以进展总览为准）
- [TASK_HUMAN_IMPORT.md](TASK_HUMAN_IMPORT.md)：任务理解分的人工盲评导入说明
- [archive/](archive/)：规格书与协议在特定提交时的快照，供实验来源哈希复核
