# E1 问题的文献依据与候选路线

检索日期：2026-10-05。以下核对了论文原文相关章节，并对照当前代码；论文报告与对Antler的建议分开陈述。本文是调研记录，不修改实验规格、门槛或停止规则，不授予新实验资格。

## 1. 先区分监测读数与注入系数

[Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/html/2507.21509)，§2.2、§3.3、附录A.3：方向从回答token的平均激活提取；监测采用最后prompt token激活在方向上的投影。作者报告投影与行为有相关性，同时指出相关性主要来自不同提示类型之间的差异，控制提示类型后相关性较弱。

我们的提取已经使用回答平均激活；方法A读取最后prompt token，并用 `solve(UUᵀ, UΔh/s)` 求对偶系数。这不是把论文方法简单做错了，但对偶系数不能未经验证就当作人格测量读数。

**建议（未执行）：** 对现有同一批经历，并列分析普通投影、对偶系数和实际行为变化，保留全部经历及两极。数学上，对偶系数δ与普通投影p满足 `p=(UUᵀ)δ`，非正交时两者符号不必一致。分析不能通过事后挑一种算法替换原正式指标，也不能凭相关性批准注入。

## 2. 让注入强度随token变化

[Steer Like the LLM: Activation Steering that Mimics Prompting](https://arxiv.org/html/2605.03907)，§3.4–3.6：提示在不同token产生的干预强度不同。PSR用当前激活预测token级强度，例如 `ReLU(h·w+b)`，结合方向注入，并以激活MSE或回答似然拟合提示干预。论文在人格等基准上报告了更好的控制/连贯性取舍。

**对本项目的意义：** 动态强度可能减少持续注入对非人格相关计算的影响，比继续加大常数系数更贴近当前质量问题。但它是新机制，须在冻结模型的前提下另行设计，通过Steer实现并重新验证；论文模型与任务不等于Qwen3-4B上的成功证据。

## 3. 用学生自己的生成轨迹蒸馏

[On-Policy Context Distillation for Language Models](https://arxiv.org/html/2602.12275)，§3：学生在不带上下文的条件下自行生成，沿学生轨迹最小化相对带上下文老师的反向KL。

当前 `antler/memory/kl_validation.py` 在固定老师回答token上计算老师到学生的正向KL。学生自由生成时可能访问不同状态，这是值得检验的训练/使用差异，并非当前实验已确认的失败原因。

**候选改动（未实施）：** 只优化δ，采用学生轨迹蒸馏。原论文主要将知识写入模型参数；迁移到仅五个参数的δ并非论文已经证明的结果，不保证解决容量限制。不能直接照搬权重更新，违反本项目冻结权重要求。

## 4. 正交化并非通用解法

[Do Personality Traits Interfere? Geometric Limitations of Steering in Large Language Models](https://arxiv.org/html/2602.15847)，§2、§4：比较软白化、Gram–Schmidt、选择性投影及硬正交化等方法。在线性重叠被去掉之后，行为串扰仍存在，部分条件流畅性降低。

因此不建议把“几何正交”当作“行为独立”的证明，也不据此重开已有剥离搜索。

## 5. 更大的方法变化：局部方向或流形

[Steering Vector Fields for Context-Aware Inference-Time Control in Large Language Models](https://arxiv.org/html/2602.01654)，§3：学习可微概念评分函数，以当前激活处的梯度决定局部注入方向，并在生成中刷新。针对静态方向在不同上下文中失配、弱效应和反向效应提供了机制思路。

[PersonaManifold: Revealing and Exploiting Curved Geometry in LLM Persona Representations](https://arxiv.org/html/2609.34571)，§3.3、§4：用局部几何与测地线路径进行人格表示插值，报告比线性插值更自然的中间状态。其特定人格数据上的15–23维估计不能直接用于断言我们的五维表示必然不够。

这两条路线改动和数据需求较大，优先级低于现有坐标解释审计。它们可能改变“固定五维线性记忆”的核心主张，需要明确登记，不能包装成原方法已成功。

## 6. 直接前作的边界

[PERSONA](https://arxiv.org/html/2602.15669)，§2.4：Persona-Flow先通过一次推理预测上下文相关的控制系数，再组合人格方向。该流程不等于Antler从经历的激活差自动恢复五维位移，也没有直接替我们证明这种逆向映射可行。

## 建议顺序

先检查现有坐标含义与行为映射；再根据证据决定是否提出固定规模的PSR式对照或学生轨迹蒸馏。完整δ能力检查只是资格前置检查，不会单独修复语义。所有新方法目前均为候选，不能声称能把11.40%提升到某个数值，也不改变原始结果或E1未完成状态。
