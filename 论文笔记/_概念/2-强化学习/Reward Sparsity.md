---
type: concept
aliases: [奖励稀疏, 奖励稀疏性, sparse reward]
---

# Reward Sparsity

## 定义
在 RL 训练中，当二元准确率指标导致同一问题组内所有正确（或所有错误）响应获得完全一致的奖励时，组内标准化产生零优势，梯度信号消失。是 GRPO 等组内标准化方法的核心痛点。

## 数学形式

当所有 $G$ 个响应 reward 相同时：

$$\tilde{A}_i = \frac{r - r}{0 + \epsilon} = 0$$

导致该问题的策略更新完全无效。

## 核心要点
1. 二元准确率奖励的固有缺陷：只有对错两档，丢失了丰富的质量信息
2. 在 GRPO 中特别严重，因为优势依赖组内方差
3. 常见解决方案包括：长度奖励增加 reward 方差、process reward 替代 outcome reward、显式惩罚建立度数梯度
4. 是 GRPO-LEAD 三项增强的共同设计动机：长度奖励增加组内方差，显式惩罚增加 reward 差距，难度重加权调节更新幅度

## 代表工作
- [[GRPO-LEAD]]: 以奖励稀疏为核心问题，提出 L/E/AD 三项增强。
- [[Process Reward]]: 用过程奖励缓解 outcome 稀疏。
- [[Reward Granularity]]: 从粒度角度研究奖励稀疏问题。

## 相关概念
- [[Group Relative Policy Optimization]]
- [[Advantage Normalization]]
- [[Outcome Reward]]
- [[Process Reward]]
- [[Reward Shaping]]
