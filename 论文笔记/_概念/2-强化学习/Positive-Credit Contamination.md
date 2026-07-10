---
type: concept
aliases: [正向信用污染, Positive-Credit Contamination]
---

# Positive-Credit Contamination

## 定义
GRPO 风格 RLVR 中的一种关键失效模式：正确轨迹中低概率的不可靠尾 token 由于 uniform credit assignment 获得与合理 token 相同的正向 advantage，导致错误行为被不自觉强化。

## 数学形式
在 GRPO 广播规则下: $\hat{A}_{i,t} = \hat{A}_i$，对所有 token $t$ 在 completion $i$ 中均相同，不论其局部可靠性。

## 核心要点
1. 根源：outcome reward 只验证最终答案，不验证每个 token 的上下文有效性
2. 加剧因素：长推理轨迹、稀疏正确动作、小 group size
3. 结果：策略逐渐偏向不合理推理模式

## 代表工作
- [[TACO]]: 专门解决此问题的方法

## 相关概念
- [[GRPO]]
- [[RLVR]]
- [[Tail-Risk Score]]
- [[Credit Assignment]]
