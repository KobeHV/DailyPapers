---
type: concept
aliases: [GPG, Group Policy Optimization, GP-CPO]
---

# GP-CPO (GPG)

## 定义
GPG (Group Policy Gradient) / GP-CPO (Group Policy-Constrained Policy Optimization) 是一类基于组内比较的 RL 优化算法，与 GRPO 共享组内标准化思想，主要区别在于使用对比学习或约束优化替代 PPO-style clipping。

## 核心要点
1. 与 GRPO 同属"组内相对策略优化"方法族
2. 不使用 PPO clipping，而采用其他正则化手段
3. 在部分基准上表现优于标准 GRPO
4. GRPO-LEAD 论文中引用为对比方法

## 代表工作
- GPG (Chu et al., 2025): "A Simple and Strong Reinforcement Learning Baseline for Model Reasoning"
- [[GRPO-LEAD]]: 引用 GPG 为其方法族的对比方法

## 相关概念
- [[Group Relative Policy Optimization]]
- [[PPO]]
- [[DAPO]]
