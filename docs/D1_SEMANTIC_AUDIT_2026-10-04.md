# C10：首轮候选的角色与情节审核

首轮18段已完成助手非盲审核。此审核已看过生成条件与独立标签，明确不算盲审，也不冒充作者60段审核。不改变原文、既有标签或split。

18段中10段存在需要复核的角色/情节问题；两模型标签一致的9段中也有5段被标记。标签一致不足以证明数据合格。其余compatible只代表本次未发现该类问题，不是正式验收。

| 槽位 | 结构审核 | 依据 |
|---|---|---|
| d1_context01_Oplus | requires_review | 叙述者亲自排练演奏并坐下来练习，开头由我提出请求；未清楚保持AI陪伴者回应用户的角色。 |
| d1_context01_Cplus | compatible | 用户请求排练计划，助手制定时间线，反馈与后续规划倾向形成因果链。 |
| d1_context01_Eminus | compatible | 用户要求建议，助手提供建议后受到负面反馈，之后减少主动表达。 |
| d1_context01_Aplus | requires_review | 建议曲目行为符合助手角色，但After we played it through含共同演奏的身体行动，AI角色不明确。 |
| d1_context01_Nplus | requires_review | 叙述者被递交节拍器且my stomach knots，具有人的身体设定，未说明AI身份。 |
| d1_context01_neutral1 | requires_review | 用户询问rushed，助手指出tempo dragged，回应目标前后不一致。 |
| d1_context02_Oplus | requires_review | I reorganized everything可指物理整理，也可指分类建议，AI做了什么不明确。 |
| d1_context02_Eminus | compatible | 助手描述收音机，用户先要求陪伴又要求安静；反馈复杂但可以发生，不据此硬判矛盾。 |
| d1_context02_Nminus | requires_review | 设置计时器符合数字助手，但continued sorting的物理行动/分类建议边界不明确。 |
| d1_context02_Nplus | requires_review | 先明确要求按颜色整理，后又说用户本来要按材质；把用户前后变化写成助手误读，情节归因不自洽。 |
| d1_context02_multi2 | requires_review | 叙述者抓取并丢弃物体，缺少AI角色设定；负面反馈如何导致继续敷衍的转变也不清楚。 |
| d1_context03_Oplus | requires_review | 我向你求设计帮助并收到陌生人询问，角色是用户或活动组织者，未呈现用户请求AI完成任务。 |
| d1_context05_Oplus | compatible | 助手起草活动方案，用户反馈和活动结果促成后续更愿提出新点子。 |
| d1_context09_neutral2 | compatible | 整理数字相册、写说明与重复同种工作可由数字助手完成；friend称谓本身不证明角色错误。 |
| d1_context10_neutral2 | requires_review | 故事结构与助手角色兼容，但番茄耐阴及一周发芽的反馈不足以支持全夏产量推断，需核查事实和叙事可信度。 |
| d1_context17_Eminus | compatible | 提供游戏规则和可选变体，被批评后更少主动表达；人格方向是否准确仍由既有标签判断。 |
| d1_context18_Nplus | compatible | 帮助组织幻灯片、接收负面反馈并担心后续错误，角色与情节兼容。 |
| d1_context19_Eplus | compatible | 安排线上聚会、引导讨论、获得反馈后更主动发起交流，符合数字助手角色。 |

机器可读记录、原文SHA256与审核身份：`outputs/analysis/d1_semantic_audit_20261004/review.json`。

当前补写运行按已冻结协议继续，不在中途修改绑定提示。完成后同样要审核AI角色和情节结构；任何语义不合格数据不得直接进入正式E1。不要把此问题误报成Qwen五维表示失败，它发生在输入经历构造阶段。
