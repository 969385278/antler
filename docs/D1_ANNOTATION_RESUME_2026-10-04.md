# C10：显式正号的语法规范化与已知回复续跑

新73段MiniMax在第20条返回`{"labels":{"O":+1,"N":-1},...}`，与提示中的+1概念一致，但不是合法JSON。原固定提示要求整数1或-1；不修改提示，也不重新调用模型挑标签。

新增默认关闭的`--allow-explicit-plus`只允许把JSON字符串之外、对象值位置的独立数值+1改为1。字符串、reason文本不变；重复键、非法维度、bool、+10、+1.0、缺失引号/标签/理由仍拒绝。原响应逐字节归档，逐条记录explicit_plus_normalized；新旧所有回复应用同一解析规则。

新增默认关闭的`--resume <failed-directory>`：先核验已失败状态、模型、厂商、接口、盲包哈希、固定提示哈希、全部归档产物、连续请求/响应前缀以及既有标签。任何请求缺少响应就拒绝自动续跑，不重发未知请求。创建新目录，逐字节复制已有请求/响应，重新解析但不得改变既有标签；仅对从未请求的后缀发起调用。相同归档加一次性claim，避免重复启动。原目录保持不变。

本次固定复用20条归档回复，保留19条旧标签，规范化第20条，最多请求剩余53条。MiniMax-M3、temperature0、max_tokens1024、thinking disabled及标注提示均不变。遇调用或其他格式失败仍停止，不自动重试。

命令：

```bash
python experiments/d1_annotate.py --config configs/main.yaml --packet outputs/main/d1_builder/ed2c320328b044d4b28f70283389c309/blind.jsonl --provider minimax --resume outputs/main/d1_annotations/minimax_2806db15d24e408c957593bc44aa51bc --allow-explicit-plus
```

这只修复数据处理和续跑，不改变生成预期、语义标准、原始回答或任何E0/E1验收门槛。新73段GLM归档保持不变，不重复请求。
