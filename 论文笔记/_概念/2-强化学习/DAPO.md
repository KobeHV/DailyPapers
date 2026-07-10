---
type: concept
aliases: [Decoupled Clip and Dynamic Sampling Policy Optimization, DAPO]
---

# DAPO

## 定义
一种 GRPO 变体，通过解耦 clip 机制和动态采样策略来改进 GRPO 训练稳定性和效率。

## 核心要点
1. 解耦的 clip-higher 机制：对正负 advantage 使用不同的 clip 阈值
2. 动态采样策略：过滤掉 group 中全部正确的样本
3. 在 reasoning RL 训练中作为重要 baseline

## 代表工作
- [[TACO]]: 对比 baseline 之一
- DAPO-Math-17K: DAPO 使用的训练数据集

## 相关概念
- [[GRPO]]
- [[PPO]]
- [[RLVR]]
