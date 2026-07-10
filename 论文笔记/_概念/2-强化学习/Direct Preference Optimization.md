---
type: concept
aliases: [DPO, 直接偏好优化]
---

# Direct Preference Optimization (DPO)

## 定义
DPO 是一种无需显式训练奖励模型的偏好优化方法，直接利用偏好数据对策略模型进行优化，将 RLHF 的奖励建模和策略优化两个阶段合并为一个过程。

## 核心要点
1. 不需要独立的奖励模型，直接从偏好对中学习
2. 在 DeepSeekMath 的统一范式分析中，DPO 被归类为"离线采样 + Rule-based 奖励 + 特定梯度系数"的方法
3. DPO 的梯度系数取决于偏好对之间的策略概率比

## 代表工作
- [[DeepSeekMath]]: 在统一范式框架中对比分析了 DPO

## 相关概念
- [[PPO]]
- [[GRPO]]
- [[RLHF]]