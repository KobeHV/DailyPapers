---
type: concept
aliases: [KL Divergence, Kullback-Leibler Divergence, 相对熵]
---

# KL 散度

## 定义
KL 散度（Kullback-Leibler Divergence）是衡量两个概率分布之间差异的非对称度量，定义为 $D_{KL}(P||Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$。

## 核心要点
1. 非对称度量：$D_{KL}(P||Q) \neq D_{KL}(Q||P)$
2. 在 RLHF/PPO/GRPO 中用作正则化项，防止策略偏离参考模型过远
3. GRPO 使用无偏 KL 估计器，直接加到 loss 中而非混入 reward 计算

## 代表工作
- [[DeepSeekMath]]: GRPO 使用无偏 KL 估计器 $\mathbb{D}_{KL}[\pi_\theta||\pi_{ref}] = \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$

## 相关概念
- [[GRPO]]
- [[PPO]]