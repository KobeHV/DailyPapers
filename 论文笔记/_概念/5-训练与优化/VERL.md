---
type: concept
aliases: [Volcano Engine Reinforcement Learning]
---

# VERL

## 定义
Volcano Engine Reinforcement Learning (VERL) 是一个开源库，提供大规模语言模型训练的 RL 算法高效实现，包括 [[PPO]]、[[GRPO]] 等。

## 核心要点
1. 专为大规模语言模型 RL 训练设计
2. 与 [[vLLM]] 集成实现高效推理和 rollout 生成
3. 支持多种 RL 算法（PPO、GRPO 等）
4. 优化了 GPU 内存和计算效率
5. 被 [[Reward Granularity in RLVR]] 等研究用作 RLVR 训练框架

## 代表工作
- [[Reward Granularity in RLVR]]: 使用 VERL + vLLM 实现 Qwen2.5-0.5B 的 GRPO 训练
- [[GRPO]] 原始实现基于 VERL 框架

## 相关概念
- [[GRPO]]
- [[PPO]]
- [[vLLM]]
- [[RLVR]]
