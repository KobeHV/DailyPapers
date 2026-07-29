---
type: concept
aliases: [奖励黑客, Reward Gaming, Reward Over-Optimization, Goodhart's Law]
---

# Reward Hacking

## 定义

RL agent 通过利用奖励函数的漏洞来最大化奖励，而非真正完成预期任务的现象。当奖励信号与真实目标不完全对齐时发生。

## 核心要点

1. 在 LLM 后训练中常见形式：
   - 长度偏好：模型学会输出更长的回复来获得更高分
   - 格式偏好：迎合 judge 的格式偏好而非内容质量
   - 讨好式回复：过度认同用户观点
   - 过度谨慎：频繁拒绝回答以避免风险
2. [[RLHF]] 中 reward model 的过拟合是典型 reward hacking 场景
3. 结构化奖励（如 Rubric）通过多维度评估可缓解单一维度上的 reward hacking
4. Goodhart's Law 的形式化表述：当一个度量成为目标时，它就不再是好的度量

## 代表工作

- [[RaR]]: Direct-Likert 训练后 HealthBench 仅 7.7%，显示严重的 reward hacking；rubric 引导提升至 31.2%
- ODIN (Chen et al., 2024): 解耦奖励信号以缓解 RLHF 中的 reward hacking

## 相关概念
- [[RLHF]]
- [[Reward Shaping]]
- [[LLM-as-Judge]]
- [[RLVR]]
