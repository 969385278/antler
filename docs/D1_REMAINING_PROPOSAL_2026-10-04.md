# D1 剩余数据缺口处理提案（已授权，执行前登记）

当前完整 E1 仍需原定 280 段经历和 60 段作者方向盲标。233 段只是暂留，不能改称完整数据。作者已明确授权固定47段补写一轮，并将60段方向复核交由助手负责。已有 dev 优化独立继续。

## 方案：再做一次固定 47 槽位补写

- 锁定已保留 233 段，文本、标签、家族、split 不变。
- 下列 47 槽位各一次新请求，沿用上轮生成模型、参数、提示与机械验收；仅改变版本号及前版哈希。总上限 47 次生成、机械合格者各两次独立标签请求，最多 94 次标注；不自动重试。
- 新请求不含旧回答、失败理由、评委分数或 Qwen 结果，不针对已测模型效果改写。
- 保留所有旧版本；新版本通过相同机械、结构和两模型目标标签规则后才能纳入。只允许单一版本，不回选被拒旧文本。
- 结构审核先封存再揭示本批模型标签；如由助手执行，明确报告已知历史背景，不冒充人类或完全盲审。
- 结束后仍有缺额则如实报告，不自动开下一轮；不减少配额、不取消中性和多目标。
- 本轮尚未发起生成或标注；执行前需准备版本兼容、来源哈希和单元测试，CPU 全套通过后才发送请求。

## 独立的人工作业要求

RESEARCH_SPEC 第 5.3 节仍要求作者本人对每维每方向各 6 条、共 60 条做隐藏目标的方向标注。已有 D5 的助手代评分授权明确针对那批评分，不能直接将 D1 助手标注写成人工验收。

推荐保留作者盲标：先导出只有随机编号和文本的固定 60 条包，作者选择 O/C/E/A/N 各 −/0/+，再计算与两模型的一致性。作者无法参与时，可以明确修订为助手辅助标签，但必须标为非人工、报告已知文本和预期的污染风险，原人工要求仍未满足，不能暗中算通过。

## 固定槽位来源

来源：`outputs/analysis/d1_replacement_selection_20261004.json`；SHA256 `7b12e43c5b2ffa1ec4c734938d05565d0527dd6661db44d1dd3b461644a6ad89`。

| 槽位 | 现有排除原因 |
|---|---|
| d1_context01_Oplus | new_structure_rejected |
| d1_context01_Aminus | new_label_rejected |
| d1_context01_Nplus | new_structure_rejected |
| d1_context01_neutral1 | new_label_rejected |
| d1_context01_neutral2 | new_label_rejected |
| d1_context02_Aminus | new_label_rejected |
| d1_context02_Aplus | new_structure_rejected |
| d1_context02_neutral2 | new_label_rejected |
| d1_context03_multi2 | new_label_rejected |
| d1_context04_Aminus | new_label_rejected |
| d1_context04_neutral2 | new_label_rejected |
| d1_context04_multi1 | new_label_rejected |
| d1_context04_multi2 | new_label_rejected |
| d1_context05_neutral2 | new_label_rejected |
| d1_context06_multi1 | new_label_rejected |
| d1_context06_multi2 | new_label_rejected |
| d1_context07_multi1 | new_label_rejected |
| d1_context07_multi2 | no_new_candidate |
| d1_context08_multi1 | new_label_rejected |
| d1_context09_neutral2 | new_label_rejected |
| d1_context09_multi1 | new_label_rejected |
| d1_context10_neutral2 | new_label_rejected |
| d1_context10_multi2 | new_label_rejected |
| d1_context12_Aminus | new_label_rejected |
| d1_context12_neutral1 | new_label_rejected |
| d1_context12_multi2 | new_label_rejected |
| d1_context13_Aminus | new_annotation_invalid |
| d1_context13_multi1 | new_label_rejected |
| d1_context13_multi2 | new_label_rejected |
| d1_context14_neutral2 | new_label_rejected |
| d1_context14_multi2 | new_label_rejected |
| d1_context15_Eminus | new_label_rejected |
| d1_context15_Aminus | new_label_rejected |
| d1_context15_multi1 | new_label_rejected |
| d1_context15_multi2 | new_label_rejected |
| d1_context16_Oplus | new_structure_rejected |
| d1_context16_multi2 | new_label_rejected |
| d1_context17_Eminus | new_label_rejected |
| d1_context17_neutral1 | new_label_rejected |
| d1_context17_neutral2 | new_label_rejected |
| d1_context18_multi2 | new_label_rejected |
| d1_context19_multi1 | new_label_rejected |
| d1_context19_multi2 | new_label_rejected |
| d1_context20_Aplus | new_structure_rejected |
| d1_context20_neutral1 | no_new_candidate |
| d1_context20_multi1 | new_label_rejected |
| d1_context20_multi2 | new_label_rejected |

## 本次授权落地

用户明确授权后执行一次第二轮，replacement_round=2、attempt=4；attempt只是版本代号。执行沿用原生成提示和参数；来源台账哈希不变。60段方向复核由助手负责，替代本轮作者亲自标注的执行要求，来源仍为非人工；human_review_complete=false仅表示未取得人类标签，不再作为等待用户打分的任务。盲性与标注者身份只能按已有来源证据报告。此变更不等于D1或E1通过。
