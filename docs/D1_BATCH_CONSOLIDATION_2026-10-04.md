# C10：跨批独立标签汇总

新增`experiments/d1_merge_batches.py --manifest <JSON>`。清单为批次对象列表，每项指定id、candidates、prepared、labels_a、labels_b，所有路径均为本地真实归档。标注必须完成，并核验其请求/响应产物哈希；逐批重新核验盲包、私有映射、输入哈希及标签完整性后才合并。

不同批次可重复使用D1B0001等盲ID，因此先按各自映射恢复原槽位，不能直接拼接标签。跨批原槽位或规范化文本重复一律报错，包括已拒收故事也不得用后续批次静默覆盖。原split继承冻结make_plan，模型和标注者顺序必须相同。原始标签不重评，分歧不删除。

输出label_accepted.jsonl与label_rejected.jsonl，每条附label_batch和semantic_review=pending。名称刻意区分标签一致与正式数据验收；formal_ready、semantic_review_complete、human_review_complete保持false。报告累计缺额、各split配额、多维逐维正负数量和逐维一致性，不把总量凑齐视为自动通过。

6个新测试覆盖盲ID碰撞、跨批重复槽位、分歧保留、映射/split/模型/哈希错误。此新增工具未修改正在执行的生成/标注入口。

## 当前实际数据

第二次生成在77个请求后因ProxyError按规则停止，新增73段机械合格，185槽位未请求。237产物哈希核验通过。原18段与新增73段必须分别保留来源。

新73段GLM已全部完成，MiniMax已完成19段后，第20段返回显式+1造成严格JSON解析失败。原文明确给出O:+1、N:-1，标签本身不缺失。不得重新猜标签或重复请求前19段；后续仅可登记无歧义的数值语法规范化并从已归档回复继续，原始响应必须保留。当前未完成的MiniMax批次不能送入本工具或宣称本批独立标注完成。
