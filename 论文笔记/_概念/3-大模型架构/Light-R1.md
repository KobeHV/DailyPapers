---
type: concept
aliases: [Light-R1-14B-DS, LightR1]
---

# Light-R1

## 定义
Light-R1 是一个 14B 规模的数学推理模型，基于 DeepSeek-R1-Distilled-Qwen-14B，使用 SFT + GRPO 两阶段训练（含余弦长度奖励）。是 GRPO-LEAD 论文的主要对比基线。

## 数学形式

训练策略：
1. SFT on ~3k 高难度数学题
2. GRPO with cosine-based length reward on 精选数学题（3 epochs）

## 核心要点
1. 与 GRPO-LEAD 共享同一基座模型（DeepSeek-R1-Distilled-Qwen-14B）
2. 使用余弦长度奖励而非指数衰减
3. 在 AIME 上 Cons@32 为 0.833 (AIME24) / 0.767 (AIME25)
4. GRPO-LEAD 以更少训练步数（100 vs 3 epochs）实现匹配或超越

## 代表工作
- [[Light-R1]]: Wen et al., 2025. 14B 数学推理 SFT+GRPO 模型.
- [[GRPO-LEAD]]: Zhang & Zuo, 2025. 以 Light-R1 为对比基线.

## 相关概念
- [[Group Relative Policy Optimization]]
- [[DeepSeek-R1]]
- [[Length-Dependent Accuracy Reward]]
