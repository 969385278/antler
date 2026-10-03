# 30条任务理解盲评导入

先按`task_blind_instructions.md`填写`task_blind.csv`中的`task_human`列。评分前不查看条件映射或实验汇总。只填0–100分，保留ID、问题和回答；可以调整行序、添加备注列。

完成后在仓库根目录运行：

```bash
python experiments/e0_import_task_labels.py \
  --config configs/main.yaml \
  --csv <盲评目录>/task_blind.csv \
  --key <盲评目录>/task_blind_key.json
```

若原始`steps/`放在其他目录，添加`--source-root <包含steps的目录>`。导入会核对原始问题、回答、实验条件、样本下标及评委分；原始文件缺失或不一致时停止，不凭ID猜测。

结果保存到配置输出目录的`human_task/<运行编号>/`：

- `agreement.json`：有效配对数、缺分ID、Spearman相关和任务评委校验状态。
- `summary.md`：简明结论。
- `run_meta.json`：版本、输入哈希和运行状态。

原定30条全部填写且评委分完整、Spearman≥0.6才标记任务评委校验通过。空白不补零；常量评分相关性不可计算。工具只读输入，不修改作者评分，不调用模型或评委。

**范围：只覆盖任务理解评委的30条校验。** D5要求的100条人格/连贯性盲评、20条间隔至少7天复标、第二评委比较仍待完成。此工具不会宣告E0通过。
