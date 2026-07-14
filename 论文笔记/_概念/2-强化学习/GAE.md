---
name: gae
description: Generalized Advantage Estimation — 广义优势估计
metadata:
  type: reference
---

# GAE (Generalized Advantage Estimation)

## 一句话
通过 **TD($\lambda$)** 思想平衡偏差与方差，为策略梯度提供低方差优势估计。

## 公式
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

## 参数
- $\gamma$: 折扣因子（偏置-方差权衡）
- $\lambda$: GAE 参数（$\lambda=0$: TD(0) 高偏置低方差；$\lambda=1$: MC 低偏置高方差）
- 典型值: $\gamma=0.99, \lambda=0.95$

## 关联
- [[PPO]]: 标准使用 GAE 计算优势
- [[DeepSeekMath|GRPO]]: 用组归一化替代 GAE（去价值网络）
- [[SAO]]: Skip-Observation GAE 用于多轮 Agent 任务
