---
name: rlhf
description: Reinforcement Learning from Human Feedback — 基于人类反馈的强化学习
metadata:
  type: reference
---

# RLHF (Reinforcement Learning from Human Feedback)

## 一句话
使用人类偏好数据训练奖励模型，再通过 [[PPO]] 等 RL 算法微调语言模型的对齐方法。

## 标准流程
1. **SFT**: 监督微调基础模型
2. **Reward Modeling**: 训练奖励模型拟合人类偏好
3. **RL Fine-tuning**: 用 [[PPO]] 最大化奖励同时约束 KL 散度

## 关键改进
- [[DPO]]: 跳过显式奖励模型和 RL，直接优化
- [[DeepSeekMath|GRPO]]: 去价值网络的 PPO 变体
- [[RewardAnything]]: 原则遵循奖励模型

## 代表工作
- InstructGPT (Ouyang et al., 2022): 首个大规模 RLHF 应用
- Llama 2 (Touvron et al., 2023): 开源 RLHF 复现
- [[DPO]]: 免 RL 替代方案
