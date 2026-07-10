---
type: concept
aliases: [RLHF, Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习]
---

# RLHF

## 定义
RLHF（Reinforcement Learning from Human Feedback）是一种通过人类偏好信号来微调语言模型的对齐方法，核心是用人类反馈训练奖励模型，再用强化学习优化策略模型。

## 数学形式
$$R_{\phi}(x, y) \approx \mathbb{E}_{h \sim p_{\text{human}}}[h(x, y)]$$
$$\pi_{\theta} = \arg\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)}[R_{\phi}(x, y)] - \beta \cdot \text{KL}(\pi || \pi_{\text{ref}})$$

## 核心要点
1. 三阶段流程：SFT → 奖励建模 → RL fine-tuning
2. 奖励模型在人类偏好排序上训练，通常使用 Bradley-Terry 模型
3. KL 正则化防止策略偏离参考模型太远
4. PPO 是最常用的 RL 算法，但 GRPO、RLOO 等 critic-free 方法逐渐兴起

## 代表工作
- [[InstructGPT]]: RLHF 的经典实现
- [[Llama 2]]: 大规模 RLHF 应用
- [[DPO]]: 无需显式奖励模型的替代方案
- [[GRPO]]: 无需 critic 的 group-based RL 变体

## 相关概念
- [[GRPO]]: 无 critic 的 group-based 策略优化
- [[DPO]]: 直接偏好优化
- [[PPO]]: 近端策略优化
- [[SFT]]: RLHF 的前置步骤
- [[LAT/LPA]]: 安全对齐的替代范式
