# C5 固定前缀延迟选项预测：运行前登记

## 测量范围

现有MMLU在prefill后立即读取选项logit，decode尚未直接生效。新增独立补充诊断：原canonical提示先正常prefill，然后强制一个换行token，带新建的非空KV cache做第二次forward，读取A/B/C/D logit。该次forward通过Steer的decode阶段注入。

这测量固定前缀条件下的一次有效干预，不是自然生成答案，也不覆盖长推理。保留原MMLU实现和正式门槛；补充结果必须标记`formal_gate_eligible:false`，不能凭它宣布decode整体能力保持。

## 固定矩阵

- 第21层、同一完整paired目录；batch_size=16，模型权重冻结。
- 固定参照：`outputs/main/capability/2026-10-03T052147.037310_0000_805c46bd/`；索引SHA256为`8af9e5913d145d889ac7455dee4c4d402612027abb1a93891e231b9317240ed9`，baseline为`bacc5d519ab1eac28a683eb09d5d06bc4b5021581327866b17c16a77afc4d490`。唯一原题库`test-00000-of-00001.parquet` SHA256为`74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`。
- 读取已有MMLU `sample_indices.json`的1400个原题号和原顺序，并对照原本地题库及已有基线的题号、学科、答案；不重新抽样或改选项。
- 三条件：zero、decode C−0.175、decode N+0.165；共4200个延迟选项预测，每条件题目两次forward；没有生成或评委调用。
- 首次forward同时保存无前缀的选项预测，各注入条件首次logit必须与本轮zero相同。原实现未显式传padding位置；本补充使用正确的逐行position_ids并记录，若历史canonical基线有差异如实报告，不覆盖旧结果。
- 固定前缀文本仅为`\n`。运行前检查为一个非特殊、非EOS、非答案字母token，且与ABCD连接后的token边界符合预期；失败即停止，不换前缀挑准确率。
- 每个条件、每批独立缓存。左padding的attention_mask、position_ids与cache长度必须正确；第二次forward必须被轨迹证明为decode并执行非零向量注入。

## 保存与分析

- 保存原题号、学科、问题/提示哈希、答案、token长度、prefix/ABCD IDs、向量哈希及范数、批次行映射、阶段与applied轨迹。
- 保存两次四选项logit、预测、四选项总概率质量、全词表最高概率token；所有数值有限。
- 对照zero的延迟预测报告准确率变化、对错翻转和逐题配对bootstrap区间；同时报告换行前后zero的变化。2个百分点仅作描述性参照。
- 输入哈希变化、缺失或非法logit、题号错配、无有效decode注入等保存失败状态并停止；不删题、补零或重跑挑结果。
- 三条件完成后停止，不自动加入all/prefill、新系数、新前缀或更多步数。后续采用新机制之前仍须另行锁定完整能力验证协议。

本轮能力条件只覆盖原幅度，不覆盖强度实验新增的中/上幅度；不能将其结果外推到更强候选。
