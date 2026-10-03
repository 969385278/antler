# C5：N剥离候选生成式能力诊断（运行前登记）

目的：旧首token选项logit没有暴露于decode-only干预，不能证明N剥离+0.20保留能力。本轮测量生成回答后的选择；只作新机制的能力诊断，不替代正式canonical MMLU验收，不改变默认all机制或E0门槛。后续正式采用需要明确协议修订。

固定第21层、N+0.20、decode-only，直接复用1080条实验的实际projected张量，不重拟合q、不改范数。来源 `outputs/main/quality_projection/2026-10-03T202637.350816_0000_38bbf8b3/injection_vectors.pt` SHA256=b20e20713aa4f506965c7e976a675d14b7700e1c61f6f8bf70ebc07d42f47510；N张量原始字节SHA256=fa2984b545fa63f488018f3a358ec0320a69e0e08092ce5252a0d7656c665617。模型/tokenizer辅助文件须与该运行来源哈希一致；不声称本轮全量重验权重。

沿用已有1400题原顺序与57学科，使用e0_delayed_capability.fixed_inputs的既定来源哈希，历史分数只用于身份校验。两臂zero/N_projected，同期相同提示、192自由token、贪心、batch16、seed0。每批交替先执行zero/注入，不更改臂内题序。无强制assistant前缀，不做多次采样，不调用评委；无提示截断。先用前两道原题做两臂零张量各8token相等性预检，共4条；不将预检纳入准确率、不按结果调提示。

用户提示为原题及ABCD选项，随后固定：`Give a brief reason, then end with exactly one line in this format: Final answer: X (replace X with A, B, C, or D).` 不添加系统提示，不修改BASE_SYSTEM。

解析规则：最后非空行去首尾空白后须匹配 `Final answer: [ABCD]`（忽略大小写）。若存在与末行不同的显式 `Final answer: X` 或 `answer is X` / `correct answer is X`，按冲突计错；匹配 `I cannot/can't/will not/won't/am unable to answer/provide an answer/provide the answer` 按拒答计错。其他无合法末行均计错，不人工猜测、补答案、增加预算或重试生成。截断但合法末行按规则计分。模式只覆盖列明的拒答/冲突措辞，不声称语义解析完整。

每条保存文本、token、结束原因、答案解析及选择token位置、选择前和结束前实际注入次数；batch保存完整Steer轨迹。首token不受decode注入；未解析答案没有“选择前次数”，记null而非0。报告暴露次数分布，不能因题数足够就宣称长期暴露充分。

固定分母1400，报告两臂正确数/准确率、同题准确率下降、对变错/错变对、10000次seed0按题配对bootstrap区间、解析失败及截断。下降≤2个百分点仅作诊断参照，不新增区间硬门槛。保存全部2800条；输入变化、非有限值、轨迹异常或异常退出即停止，不自动换测法/预算。完成后不根据成绩追加条件；保留失败。
