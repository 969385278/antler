# C10：MiniMax与GLM独立盲标接口

作者指定两份独立标签来源为MiniMax与GLM，主评委DeepSeek不变。新增`experiments/d1_annotate.py`，每次只运行其中一个来源，读取同一个`blind.jsonl`，不读取候选预期、私有映射或另一模型结果。

标注提示要求判断经历结果之后可能形成的持久变化，区分“经历中做了什么”和“经历让人格朝什么方向变化”。返回五维中的正负整数及简短依据；没有方向证据时可返回空对象。固定提示见`antler/eval/episode_annotation.py`，不修改人格评分、连贯性或任务理解提示。

配置从项目`.env`或同名环境变量读取：`MINIMAX_API_KEY/BASE_URL/MODEL`与`GLM_API_KEY/BASE_URL/MODEL`。模型名和地址必须显式配置，不猜版本。入口使用chat/completions兼容接口；不确定兼容性时先单独连通性检查，不直接启动正式盲标。

每次最多280条，temperature0、max_tokens1024，逐条请求，无自动重试。记录无鉴权头的请求体、响应体、请求/响应哈希和实际返回模型，禁止接受截断响应、模型身份变化、重复JSON键、非法维度或bool冒充整数。失败保留现场及已完成标签，不通过重抽回答挑选有利标签。独立请求使用相同题序不妨碍输入隔离；模型之间没有共享对话历史。

输出`labels.json`兼容D1合并工具，记录请求模型、厂商、盲标包哈希及逐条标签。密钥不写入run_meta、请求归档或错误消息。该入口不代表数据已通过语义审核或60段人工盲审，也不证明两个厂商的误差统计独立。

命令：`python experiments/d1_annotate.py --config configs/main.yaml --packet <blind.jsonl> --provider minimax`，GLM单独替换provider为glm。单元测试使用模拟HTTP响应。

## 配置和真实连通性检查

作者提供密钥并授权配置后，已写入Git忽略的本地`.env`，不复制进实验元数据。选用官方文档列出的非预览模型MiniMax-M3与glm-5，分别使用`https://api.minimaxi.com/v1`和`https://open.bigmodel.cn/api/paas/v4`。两次独立最小请求均HTTP200、finish_reason=stop，返回模型与请求一致。此检查只要求回复OK，不涉及D1或任何预期标签。

已验证的请求设置为temperature0、thinking.type=disabled；正式标注沿用，并将max_tokens从连通性检查的32固定为1024。MiniMax较旧M2系列不能关闭thinking，本入口当前设置不声称兼容所有版本，不自动切换模型。D1正式标注尚未开始。

官方接口依据：https://platform.minimaxi.com/docs/api-reference/text-openai-api 与 https://docs.bigmodel.cn/cn/guide/models/text/glm-5 。连接结果（不含密钥）保存于`outputs/analysis/annotation_connection_check.json`。
