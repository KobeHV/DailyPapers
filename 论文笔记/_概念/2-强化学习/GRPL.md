---
type: concept
aliases: [Group Relative Preference Learning, 分组相对偏好学习]
---

# GRPL

## 定义
Group Relative Preference Learning (GRPL) 是一种基于 [[GRPO]] 的强化学习训练方法，专门用于训练[[Reward Model|奖励模型]]遵循自然语言原则进行评估。与传统[[RLHF]]中学习隐式偏好不同，GRPL 将奖励模型视为策略 $\pi_\theta$，通过强化学习优化其生成符合给定原则的评估输出的能力。

## 数学形式

$$
J_{GRPL}(\theta) = \mathbb{E}_{q, \{o_i\} \sim \pi_{old}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t} \min\left(r_t(\theta)\hat{A}_{i,t},\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]
$$

## 核心要点
1. **策略视角**: 将奖励模型的评估过程视为策略 $\pi_\theta$ 生成评估输出 $o_i$（推理+评分+排序）
2. **相对偏好**: 对比 GRPO 优化数学推理，GRPL 优化的是评估质量，使用相对偏好奖励而非绝对正确性
3. **两阶段奖励**: 总奖励 = 格式奖励 + 准确度奖励，分别评估输出格式质量和与 ground truth 的对齐程度
4. **Listwise 训练**: 同时处理 $k$ 个候选回答的排序，比 pairwise 训练更高效

## 代表工作
- [[RewardAnything]]: 首次提出 GRPL，实现原则遵循的奖励模型

## 相关概念
- [[GRPO]]
- [[PPO]]
- [[RLHF]]
- [[Reward Model]]