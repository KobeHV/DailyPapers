---
type: concept
aliases: [On-Policy Reinforcement Learning, 在线策略强化学习, On-Policy]
---

# On-Policy RL

## 定义

强化学习中策略 $\pi_\theta$ 用于采样行为数据，同时该数据用于更新同一策略 $\pi_\theta$ 的训练范式。与 [[Off-Policy]] 相对，on-policy 要求每次策略更新后重新采样数据。

## 数学形式

On-policy 策略梯度的一般形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]
$$

其中 $A^{\pi_\theta}$ 是使用当前策略 $\pi_\theta$ 估计的优势函数。

## 核心要点

1. 采样策略与优化策略是同一个，避免了 off-policy 中的分布偏移问题
2. 经典算法包括 PPO、TRPO、A2C/A3C
3. 在大模型后训练中，GRPO 是 on-policy 的代表算法：每轮用当前 policy 采样 rollout，计算 reward 后更新
4. 优势：训练稳定性好，reward 信号与当前模型行为一致
5. 劣势：样本效率低（每轮需重新采样），计算开销大

## 代表工作

- [[PPO]]: 最广泛使用的 on-policy RL 算法
- [[GRPO]]: 用 group relative advantage 替代 value network，推动大模型 RL 训练
- [[RaR]]: 用 rubric 奖励 + GRPO 实现 on-policy 训练

## 相关概念
- [[Off-Policy]]
- [[PPO]]
- [[GRPO]]
- [[GAE]]
- [[TRPO]]
