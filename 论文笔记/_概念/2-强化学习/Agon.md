---
tags: [concept, rl, competitive-rl, multi-model, reasoning]
created: 2026-07-09
---

# Agon

## 定义

Agon（希腊语 agon，意为"竞赛/较量"）是一种跨模型竞争强化学习框架，由 Vladislav Beliaev 于 2026 年提出。核心思想是让**两个能力相当但行为不同的模型在 RL 训练中互为隐式评分器**，通过奖励"击败对手"来隐式评判推理质量，无需 process reward model 或人工标注。

## 核心机制

1. **Draft-and-Challenge**: 一个模型起草解答，另一个阅读其摘要并尝试超越
2. **Conversion Bonus**: 在对手失败时正确解题获得额外奖励
3. **Role Rotation**: 每个 optimizer step 后角色互换，两者共同提升
4. **Implicit Grading**: 推理质量通过竞争结果隐式评分，无过程标签

## 关键结果

- DeepMath-hard (Qwen3-0.6B): pass@1 = 61 (vs GRPO 30, Zero-shot 23)
- 达到约 2x GRPO pass@1，是未训练 MoA 增益的约 8x
- 涌现推理链缩短 (3.5k vs GRPO 8.1k tokens)
- 跨模型家族 (Qwen3.5, Gemma 4) 和跨领域 (CodeContests) 成立

## 与相关概念的关系

- 建立在 [[GRPO]] 的 group-relative advantage 框架之上
- 区别于 [[Self-Play]]：使用两个不同的模型而非同一模型的不同版本
- 区别于 [[Mixture-of-Agents]]：竞争性而非合作性，且在 RL 训练中应用

## 主论文

- [[Agon]]: Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning (Beliaev, 2026)
