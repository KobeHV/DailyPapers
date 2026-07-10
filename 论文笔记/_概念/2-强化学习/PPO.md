---
type: concept
aliases: [Proximal Policy Optimization, PPO]
---

# PPO

## 定义
一种基于策略梯度的强化学习算法，通过 clipped surrogate objective 限制每次更新的步长，防止策略突变导致训练崩溃。

## 核心要点
1. 使用 clipped probability ratio $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$ 限制策略更新幅度
2. 支持 Actor-Critic 架构，通常配合 GAE (Generalized Advantage Estimation) 使用
3. 在 LLM RL 训练中被广泛采用（RLHF、RLVR）

## 代表工作
- [[GRPO]]: 移除 critic 的 PPO 变体
- [[TACO]]: 在 GRPO 基础上增加 token-level credit 校准

## 相关概念
- [[GRPO]]
- [[RLVR]]
- [[Credit Assignment]]
- [[Advantage Estimation]]
