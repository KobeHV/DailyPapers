---
type: concept
aliases: [难度感知优势重加权, 难度加权优势, difficulty-weighted advantage]
---

# Difficulty-Aware Advantage Reweighting

## 定义
一种 RL 训练策略，根据问题难度动态调整策略更新幅度：对难题的正确回答给予更大的正优势权重，对简单题的错误回答施加更大的惩罚。通过放大难题的学习信号提升模型泛化能力。

## 数学形式

**难度估计**（经验正确率）：

$$\rho_q = \frac{\text{number of correct responses for } q}{\text{total number of responses for } q}$$

低 $\rho_q$ = 高难度。

**Logistic 重加权因子**：

$$w(\rho_q) = A + \frac{B - A}{1 + \exp[k(\rho_q - \rho_0)]}$$

典型参数：$A=0.4, B=1.5, \rho_0=0.75, k=10$。

**难度感知优势**：

$$A_i' = \tilde{A}_i \cdot \begin{cases} w(\rho_q), & \text{if } \tilde{A}_i > 0 \\ w(1 - \rho_q), & \text{if } \tilde{A}_i \leq 0 \end{cases}$$

## 核心要点
1. 无需人工标注难度，完全从模型实时采样的组内正确率推断
2. Logistic 函数确保在 $\rho_0$ 附近权重急剧变化，对中间难度问题最为敏感
3. 难题的稀有正确回答权重可达 $w \to B$（如 1.5x），简单题的频繁错误回答也获得 $w \to B$ 的大惩罚
4. 与长度奖励互补：长度奖励关注输出简洁性，优势重加权关注问题难度分配

## 代表工作
- [[GRPO-LEAD]]: Zhang & Zuo, 2025. 首次在 GRPO 中引入难度感知优势重加权。

## 相关概念
- [[Group Relative Policy Optimization]]
- [[Advantage Normalization]]
- [[Reward Sparsity]]
- [[Importance Sampling]]
