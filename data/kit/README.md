# 鹿角记忆 · 数据包说明

这个包里是从 GitHub 公开仓库整理出来、已经检查过的数据。HuggingFace 上的数据（MMLU、UltraChat、TRAIT）和模型，用 `scripts/download_on_server.sh` 在服务器上下载。

## 目录与用途

| 目录 | 内容 | 来源 | 在实验中的用途 |
|-|-|-|-|
| `trait_extract/` | 17 个人格极点，每个含 5 对正负系统提示 + 20 道题 + 评分提示 | PERSONA 仓库 | **提取人格向量**（E0），以及写入经历时的探针集 P |
| `trait_eval/` | 同样 17 个极点，另外 20 道题 | PERSONA 仓库 | **测量人格分数**（E0 校准、E1–E4） |
| `bfi/` | BFI-44 的 44 道题 + PERSONA 的评测脚本 | PERSONA 仓库 | 标准大五测量，可和 PERSONA 论文的数字对比 |
| `personality_bench/` | 五个维度各约 90 道情境题（共 459 题） | NPTI / PersonalityBench | **第三方外部验证** |
| `persona_evolve/` | 100 个多轮对话会话（每个 8 个情境）及评测版本 | PERSONA 仓库 | **和 PERSONA-Flow 正面对比**（E4） |
| `life_stream_reference/` | CLS 论文的 1000 段人物人生经历 + 数据构建代码 | CLS 论文仓库 | 参考它的流程，生成你自己的“AI 人生经历流” |

## 已经检查过的事项

- `trait_extract/` 和 `trait_eval/` 的题目**完全不重叠**（每个极点 20 + 20，交集为 0），两者的系统提示相同。可以放心一个用来提取、一个用来评测。
- 注意：PERSONA 仓库里还有一个 `trait_data_big_five/` 目录，每个极点 40 道题，**其中 20 道和评测集重复**，而且缺少 compassionate。**不要用那个目录做提取**，本包已经换成了干净的版本。
- `bfi/BFI.json` 不是 UTF-8 编码，读取时用 `open(path, 'rb').read().decode('latin-1')`。
- `personality_bench/*.json` 是每行一个 JSON 的格式（JSONL），不能直接 `json.load`，要逐行读。

## 17 个极点和大五的对应

| 维度 | 高极 | 低极 |
|-|-|-|
| 开放性 O | inventive | consistent |
| 尽责性 C | dependable | careless |
| 外向性 E | outgoing | solitary |
| 宜人性 A | compassionate | self-interested |
| 神经质 N | nervous | calm |

另外 7 个（aloof、antagonistic、challenging、competitive、cooperative、disciplined、secure）是 PERSONA 额外提供的细分特质，第一阶段可以先不用。

## 还需要你自己准备的

1. **TRAIT**：在 HuggingFace 上同意使用条件（mirlab/TRAIT），拿到 token 后运行下载脚本。
2. **评测/生成用的大模型 API**：用于评委打分和生成经历数据。
3. **自己构建的数据**（后续阶段）：AI 人生经历流、交互经历组、双胞胎经历、50–100 条人工评分。

## 许可

PERSONA 仓库为 MIT 许可（见 `LICENSE_persona`）。使用这些数据发表论文时，请引用原论文：PERSONA（Feng et al., 2026）、NPTI（Deng et al., ICLR 2025）、CLS（Pan et al., 2026）、Persona Vectors（Chen et al., 2025）。
