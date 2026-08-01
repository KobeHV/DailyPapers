---
type: concept
aliases: [优势归一化, 组内标准化, group-relative normalization]
---

# Advantage Normalization

## 定义
在 GRPO 类算法中，对同一问题 $q$ 的多条采样响应的 reward 进行组内 Z-score 标准化，得到标准化优势 $\tilde{A}_i$ 用于策略梯度更新。消除了不同问题的 reward scale 差异，使梯度更新更为一致。

## 数学形式

$$\tilde{A}_i = \frac{R(o_i|q) - \mu_q}{\sigma_q + \epsilon}$$

其中 $\mu_q$ 和 $\sigma_q$ 是问题 $q$ 的组内 reward 均值和标准差。

## 核心要点
1. 是 GRPO 替代 critic 网络的核心机制
2. 依赖 group size $G \ge 2$，$G$ 过小导致标准差估计不稳定
3. 归一化后正优势表示"优于组内平均"，负优势表示"劣于组内平均"
4. 在难度感知优势重加权中，标准化优势作为基础被进一步调制

## 代表工作
- [[DeepSeekMath]]: Shao et al., 2024. 首次提出 GRPO 及其组内标准化优势。
- [[GRPO-LEAD]]: Zhang & Zuo, 2025. 在此基础上的难度加权增强。

## 相关概念
- [[Group Relative Policy Optimization]]
- [[PPO]]
- [[Reward Sparsity]]
- [[Reward Shaping]]
