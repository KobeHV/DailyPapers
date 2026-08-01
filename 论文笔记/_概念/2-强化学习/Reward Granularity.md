---
type: concept
aliases: [奖励粒度, reward granularity]
---

# Reward Granularity

## 定义
在 [[RLVR]] 中，奖励信号的分发粒度——从仅评估最终答案的稀疏结果级奖励，到评估每个中间步骤的密集过程级奖励。

## 数学形式
$$R = \lambda \cdot R_{\text{process}} + (1 - \lambda) \cdot R_{\text{outcome}}$$

其中 $\lambda$ 控制过程奖励的相对权重，$\lambda=1$ 为纯过程奖励，$\lambda=0$ 为纯结果奖励。

## 核心要点
1. 奖励粒度是 [[RLVR]] 的一等设计决策，直接影响推理质量和最终准确率
2. 过程级奖励提供密集监督，改善推理忠实度但可能导致冗长推理链
3. 结果级奖励高效但让推理过程不透明且容易出错
4. 混合奖励可实现两者间的平衡，但权重不当（如极低过程权重）可能引入冲突优化信号
5. 奖励粒度的最优选择依赖于模型规模和任务特性

## 代表工作
- [[Reward Granularity in RLVR]]: 系统比较五种奖励条件在小型语言模型数学推理上的效果，证明过程监督在 Qwen2.5-0.5B + GSM8K 上比结果监督高出近 10 个百分点

## 相关概念
- [[Process Reward]]
- [[Outcome Reward]]
- [[RLVR]]
- [[GRPO]]
- [[Chain-of-Thought]]
