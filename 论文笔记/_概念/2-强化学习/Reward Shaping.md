---
type: concept
aliases: [奖励塑形, Reward Engineering, 奖励设计]
---

# Reward Shaping

## 定义

通过引入额外的奖励信号来指导 RL agent 学习的技术，核心目标是在不改变最优策略的前提下加速训练或提升最终性能。

## 数学形式

基于势能的 reward shaping (Ng et al., 1999)：

$$
R'(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)
$$

其中 $\Phi$ 为势能函数，保证最优策略不变性。

## 核心要点

1. 在稀疏奖励环境中提供密集训练信号
2. 在大模型 RL 训练中，reward shaping 体现为设计不同粒度的奖励：二元验证（RLVR）、评分量表（RaR）、process reward
3. 潜在风险：shaping 不当可能引入偏差甚至 [[Reward Hacking]]
4. Rubric-based 奖励可视为一种结构化、可解释的 reward shaping

## 代表工作

- [[RaR]]: 将评分量表转化为结构化的 reward shaping 信号
- [[RLVR]]: 使用程序化验证函数作为奖励
- Process Reward Model: 在中间推理步骤提供密集奖励

## 相关概念
- [[Reward Hacking]]
- [[RLVR]]
- [[RLHF]]
- [[GRPO]]
