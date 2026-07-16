---
type: concept
aliases: [GRPO, 组相对策略优化]
---

# Group Relative Policy Optimization

## 定义
一种强化学习策略优化算法，通过在同一 prompt 下采样一组响应并计算组内相对优势来优化策略，无需独立的 critic/value 网络。

## 数学形式

组内相对优势：

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

策略梯度：

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A_i\right)\right]$$

## 核心要点
1. 无需训练单独的 critic/value 网络，节省显存和计算
2. 组内归一化消除了 reward scale 的影响
3. DeepSeek-R1 和 DeepSeek-V系列中广泛使用
4. 适用于数学、编程、指令遵循等有明确 reward 信号的任务

## 代表工作
- [[DeepSeek-R1]]: 使用 GRPO 训练推理模型
- [[DeepSeek-V3]]: 后训练阶段使用 GRPO
- [[DeepSeek-V4]]: 专家训练阶段使用 GRPO 进行 RL 优化

## 相关概念
- [[Proximal Policy Optimization]]
- [[Direct Preference Optimization]]
- [[Reinforcement Learning from Human Feedback]]
- [[On-Policy Distillation]]
