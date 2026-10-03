# D5：100条人格/连贯性盲评与20条复标

工具处理已有回答，不生成回答，不调用评委，不填写作者评分。正式批次需在方案锁定后选定来源；探索批次不能直接变成正式E0证据。

所有命令在仓库根目录运行，结果写入配置输出目录的`human_d5/<运行编号>/`，包含`run_meta.json`。

## 1. 导出首次盲表

```bash
python experiments/e0_blind.py --config configs/main.yaml --action export \
  --samples <完整测量的calibration_judge_samples.jsonl> <C17的extract_coherence_samples.jsonl> \
  --seed 0
```

可传多个JSONL。生成测量按维度×系数分层，提取回答按维度×pos/neg分别分层（提取没有注入系数）。五维必须齐全，各层至少一条；每轮各层抽一条，余下容量不足时向其他层分配，直至100条。重复目标/问题/回答去重，数量不足或无法覆盖某层时停止。

作者只打开新目录下`bundle/instructions.md`和`bundle/blind.csv`。`target`是要评价的特质，不是干预条件。表中隐藏层号、系数、种子、来源、评委分，两个评分列均为空。

**首次及复标全部完成前，不查看任何private文件或一致性结果。** 私有映射包含来源与评委分；这些文件供程序核验。

## 2. 完整标注后导入并开始计时

```bash
python experiments/e0_blind.py --config configs/main.yaml --action import \
  --bundle <首次导出的bundle目录> --csv <已填写的blind.csv>
```

可以调整行顺序、添加备注，但不能改ID、target、问题或回答。缺分不会补零，报告有效配对数量；完整100条、两指标与评委Spearman均≥0.7才标记该相关性条件满足。工具始终不宣告E0通过。

首次完整导入时封存作者评分并记下当前时间。再次导入相同分数不重置时间；修改已封存分数会被拒绝，避免复标使用不同的首次基准。保留源文件，导入会校验哈希及映射是否对应原始回答。

## 3. 至少7天后导出复标

```bash
python experiments/e0_blind.py --config configs/main.yaml --action repeat-export \
  --bundle <首次导出的bundle目录>
```

20条复标身份在首次抽样时固定。程序从首次完整导入开始计时，未满7天拒绝导出。新目录的`repeat/repeat.csv`使用新ID、打乱顺序、不显示旧分。作者只填写这个表。

## 4. 导入复标

```bash
python experiments/e0_blind.py --config configs/main.yaml --action repeat-import \
  --repeat-key <复标目录>/private_repeat_key.json --csv <已填写的repeat.csv>
```

报告作者前后两次人格和连贯性评分的Spearman、有效配对数量以及复标表的导出间隔。导入再次检查7天间隔、首次封存与复标样本身份。

## 范围

此流程实现C9的抽样、首次导入和复标一致性。第二厂商评委的评分与三方比较尚未接入；人工真实完成、正式test来源及完整E0门槛仍需另外核验。程序只能记录操作时间，不能证明作者实际开始标注的时间或未看过私有文件。
