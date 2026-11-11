# 实验结果与图表数据

这里保存七张展示图使用的精简数值，不包含逐条回答、评委请求响应、盲评答案映射、激活向量或模型。图表在 [`figures/`](../figures/)；所有 PNG 宽 1600 px、200 dpi，并提供字体转路径的 SVG，便于跨机器查看中文。

**E0 Strict / Practical 未通过，正式 E1 验证未完成。** 历史校准、开发集选参和固定情境小样本不能升级为正式通过。报告和原始输出均保留原样。

| 数据文件（同名 PNG / SVG） | 实验、数据来源与原始运行 | 对应报告 |
|---|---|---|
| [`e0_calibration_curves.json`](e0_calibration_curves.json) | E0 第16层；`outputs/main/analysis/2026-10-03T051134.770522_0000_4a1b7c43/curves.json`；上游 `outputs/analysis/followon_1b7c4c1/oea_bipolar/` 与 `oea_pole/`；完整展示已有 O/E/A 两极曲线，源文件不含 C/N | [`RESEARCH_SPEC.md`](../docs/RESEARCH_SPEC.md) 第8节及附录B历史记录；具体数值以封存曲线为来源 |
| [`e0_crosstalk.json`](e0_crosstalk.json) | E0 第16层单极；`outputs/analysis/crosstalk_7924314/matrix/matrix.json`，原运行 `outputs/main/crosstalk_7924314/matrix`，源码 `1b7c4c1`；行=测量、列=注入 | [`RESEARCH_SPEC.md`](../docs/RESEARCH_SPEC.md) 第8.3节串扰定义；具体数值来自封存矩阵，不套用为新机制结果 |
| [`e0_tradeoff.json`](e0_tradeoff.json) | E0 同期 C 六臂对照（图中省略共同zero臂）；本地 `outputs/analysis/prompt_persona_20261004/run/`；原运行 `outputs/main/prompt_persona/d09a117a119541a2985f54a77c8e4391/`；图中数字直接读取报告表格 | [`E0_ROUND_CLOSEOUT_2026-10-04.md`](../docs/E0_ROUND_CLOSEOUT_2026-10-04.md)「本轮同期 C 结果」 |
| [`e1_kl_explained.json`](e1_kl_explained.json) | E1 dev 90条件；`outputs/analysis/e1_dev_terminal_20261004/run/results.json` 与 `selection.json`；原运行 `outputs/main/e1_kl_dev/31f70d1cb5874f7f9b3791f99eaa6d6a/`；按封存规则选空间各自学习率，保留30个逐经历统计值 | [`E1_PROGRESS_AND_ISSUES.md`](../docs/E1_PROGRESS_AND_ISSUES.md)、[`E1_PROBE_REPEAT_RESULTS_2026-10-04.md`](../docs/E1_PROBE_REPEAT_RESULTS_2026-10-04.md) |
| [`e1_twin_gate_forest.json`](e1_twin_gate_forest.json) | E1 六情境上下文闸门；`outputs/analysis/e1_twin_context_gate_20261005/summary_final.json`；原运行 `9c008133e6e94c7d91b0ff2d5a0af44f`；报告两位小数与原汇总交叉核对 | [`E1_TWIN_CONTEXT_GATE_RESULTS_2026-10-05.md`](../docs/E1_TWIN_CONTEXT_GATE_RESULTS_2026-10-05.md) |
| [`e1_state_geometry.json`](e1_state_geometry.json) | E1 第16层内部状态；`outputs/analysis/e1_twin_context_gate_20261005/outputs/main/e1_twin_state_extract/03e4e1c7f1d94fe0a31c87e623abf356/summary.json`；原运行 `03e4e1c7f1d94fe0a31c87e623abf356`；仅导出范数，不导出向量 | [`E1_TWIN_STATE_EXTRACTION_RESULTS_2026-10-05.md`](../docs/E1_TWIN_STATE_EXTRACTION_RESULTS_2026-10-05.md) |
| [`e1_l21_pilot.json`](e1_l21_pilot.json) | E1 第21层风险小样本；`outputs/analysis/e1_twin_context_gate_20261005/l21_pilot_a351909f.tar.gz`；原运行 `a351909f6f394034b5402398282a9f0c`；数值和区间来自报告，均值与归档交叉核对 | [`E1_L21_SMALL_PILOT_RESULTS_2026-10-05.md`](../docs/E1_L21_SMALL_PILOT_RESULTS_2026-10-05.md) |
| [`extraction_audit.json`](extraction_audit.json) | 提取时对原始统计与报告的差异清单；不修改历史结果；当前未发现超出报告舍入精度的差异 | 对照以上报告 |

每个图表数据文件附来源路径、来源文件 SHA256、原始运行标识和证据边界。哈希用于固定本次展示来源；压缩包内文件只读，不解压到仓库。

## 主要结果与边界

- 校准：历史 O/E/A 人格读数随注入变化，同时存在端点饱和；不据此宣称五维正式校准通过。
- 串扰：历史五维矩阵存在明显非对角变化；不同注入系数与量表端点使跨列比较需谨慎。
- 质量权衡：冻结 C 剥离版人格增益 36.67，95% CI [22.39, 51.78]，四项质量损失均超过10；Prompt强度未匹配，Strict失败、Practical未证实。
- KL：人格／固定随机五维／全维均值为11.40%／2.35%／36.34%；点为10段经历各自统计，横线为报告均值。单随机基、dev选参，不推断正式显著性；有限全维优化不是理论上限。
- 上下文闸门：六情境1个已建立、2个部分证据、3个未建立；误差线为十题配对bootstrap区间，不能解释为跨情境泛化。
- 状态几何：d+与d−余弦0.968；交互五维重建只保留0.5499%平方范数。该量不是KL解释率、解释方差或记忆保留率；固定层／位置／基底的诊断不能证明所有压缩失败。
- L21：full高极更接近对应上下文，低极与五维证据仍有限；纯反馈解释未排除。区间仅覆盖五题；一个full低极任务条件损失14分超限，能力只完成200题风险筛查。

## 复现

需要 Python、NumPy、matplotlib 和 Microsoft YaHei 字体。Windows 系统字体可直接使用；其他机器需先安装该字体，脚本在字体不可用时明确报错。SVG内嵌字形路径，不依赖阅读者安装字体。

```powershell
# 仅从已提交的小数据重画，不需要outputs、模型或网络
python scripts/make_figures.py

# 仅数据保管机器：从既有本地归档和报告重新提取，再重画
python scripts/extract_results.py
python scripts/make_figures.py
```

提取脚本只写 `results/`，画图脚本只读取 `results/` 并写 `figures/`；不运行实验、不调用评委、不加载模型。现有源汇总没有区间的曲线、串扰及范数图不补造区间；质量权衡只给增益原95% CI，质量损失明确标为点估计。报告已有区间全部保留，未重新抽样以追求结论。
