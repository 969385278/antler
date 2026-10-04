# C10：固定47槽位补写与60段助手方向复核收尾

更新：2026-10-04 13:59 北京时间。

## 结论

用户明确授权本轮固定47个缺额各补写一次，并将60段方向盲标交由助手负责。本轮已完成，D1从233增至254段，仍缺26段，不自动追加下一轮。D1与完整E1均未正式验收。

## 47槽位的完整结果

- 原233条文本、标签、家族和split逐条保持不变。
- MiniMax-M3按原提示与参数生成47次，45条机械合格，2条词数不合格。
- 先封存45条匿名结构审核：40 compatible、2 exclude、3 requires_review；不根据随后标签翻案。
- GLM 45次标注全部有效；MiniMax第8次回复包含非法JSON控制字符，保留为无效、不修复、不重问。核验8个请求与响应完整后，只请求剩余37条；总45次，其中44条有效。
- 最终互斥分类：新增21、目标标签不符20、标签符合但结构排除3、无效标注1、机械失败2。5条结构疑点中另2条同时不满足标签要求，归入标签失败，不能重复计数。
- 合计254；train/dev/test = 155/50/49。原定280及168/56/56配额保持不变。
- 两模型44对有效标签的一致性按原规则计算，完整分维结果在selection.json。非法响应不当中性，不进入一致性分母。

## 60段方向复核

沿用已有固定60条助手复核，不重复标注挑选更好分数。已重算核验抽样、文本、映射和统计：预定目标方向60/60；与MiniMax逐维一致87.3%、κ约0.710，与GLM一致93.7%、κ约0.839。

该样本来自已经通过两模型筛选的233条，不能估计筛选前准确率，也不能证明Qwen会产生对应变化。标注者身份和隔离过程来自原记录自述，本轮无供应商响应/执行轨迹独立确认。明确报告为已有AI复核；不冒充人工或新增独立复标。用户已替代本轮作者亲自标注要求，不再等待用户完成60条；human_review_complete=false仅如实表示没有人类标签。

## 可复核产物

- 选择台账：`outputs/analysis/d1_remaining_round2_20261004/selection.json`
- SHA256：`3923c0aceee07678efd658c74f702cd038637c3ae8bb774d4e352da197ef5e1d`
- 生成：`outputs/main/d1_replacement/f808364ac21e48e3ad91cddd4249acac`
- 匿名包：`outputs/main/d1_builder/139e794ca0874af68661eac1d79c9ccb`
- 结构封存、未请求后缀登记、重算脚本与来源哈希均在上述analysis目录。
- 60条来源：`outputs/main/d1_direction_review/v1_20261004`；重算审计：`outputs/analysis/d1_direction_review_audit_20261004.json`。

## 工程与后续边界

CPU全套995 passed，92.14秒，无模型下载。代码提交88915a0、d3825c1、2baa50f；服务器保持abd18f0，本轮没有部署或重启GPU实验。

最新观察：E1原PID142121存活，日志完成35/90拟合；最后两行为Cplus/full_lr0与full_lr1。此为运行日志快照，未做全量终态审计，不提前选参。新增数据未插入正在运行的固定dev网格。

剩余26缺额如实保留。本轮数据补写完成不等于正式E1完成；稳定性、正式数据验收、独立test和行为验证仍有工作，E1行为白名单仍为空。

## 未填槽位

- d1_context01_neutral2
- d1_context02_Aminus
- d1_context02_neutral2
- d1_context03_multi2
- d1_context04_neutral2
- d1_context04_multi1
- d1_context04_multi2
- d1_context05_neutral2
- d1_context06_multi2
- d1_context07_multi1
- d1_context07_multi2
- d1_context08_multi1
- d1_context09_multi1
- d1_context10_neutral2
- d1_context10_multi2
- d1_context13_Aminus
- d1_context13_multi1
- d1_context14_multi2
- d1_context15_Aminus
- d1_context15_multi1
- d1_context17_Eminus
- d1_context17_neutral1
- d1_context17_neutral2
- d1_context18_multi2
- d1_context19_multi1
- d1_context20_multi1
