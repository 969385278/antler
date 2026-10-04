# C10：剩余未请求槽位继续生成登记

第二次补写在77个请求后因ProxyError停止，185槽位未发送。已生成/标注的91段均保留；不在本轮重写标签分歧、结构存疑或机械失败故事，也不重试最后那个结果未知的请求。

新增默认关闭的`--continue-from`只读取封存`unattempted.json`，验证其恰为原request_order在completed_requests之后的后缀；已请求槽位不再请求。核验所有产物哈希、原始来源、切分、已有故事不变，独立claim防重复。新旧故事分别导出，后续只给新故事标注。

本次固定185个槽位，沿用原seed1请求顺序、原slot/group/split、MiniMax-M3、temperature .8、max_tokens512、thinking disabled、串行间隔5秒、每槽位一次，接口错误立即停止。每个槽位仍是第二次尝试，而非把先前失败槽位再抽一次。

根据已完成的角色/情节审核，显式启用`--digital-role`：叙述者是通过文字或线上工具协助用户的数字AI，用户提出请求，用户执行实体动作并反馈；经历后的变化属于AI。不得写AI的人类身体、亲戚、同学等身份，不得互换用户/助手角色。仅对尚未请求故事追加这条生成限制，原有故事不改写。保持50–65词生成目标和40–80词验收。此修订不使用Qwen实验结果、δ、KL或行为分数。

命令：

```bash
python experiments/d1_recover.py --config configs/main.yaml --source outputs/main/d1_generation/0e018efcbb8c4cd2b185214418a69c86 --continue-from outputs/main/d1_recovery/a27afff7914a4ec6bf13cfbf42ee70e7 --digital-role
```

每批生成提示版本存在差异，后续必须报告批次组成并检查是否与split/方向混淆，不能称全库同分布。原正式D1配额、独立标签、语义检查与60段盲审要求不变；本次完成不自动代表正式数据就绪。
