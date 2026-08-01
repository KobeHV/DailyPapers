---
type: concept
aliases: [DeepSeek-R1-0528-Qwen3-8B]
---

# DeepSeek-R1

## 定义
DeepSeek 推出的推理专用 LLM 系列，采用 GRPO 训练和冷启动策略，在 `<think>` 标签内生成显式推理轨迹（Chain-of-Thought），在数学和编程任务上表现出色。

## 核心要点
1. 采用多阶段训练：冷启动 SFT → RL → 拒绝采样 → 全场景 RL
2. 蒸馏版本（R1-Distill）将推理能力迁移到更小的模型
3. 其 `<think>` 推理机制是 Thinking Reward 等方法的操作基础

## 代表工作
- DeepSeek-R1 (2025): 开源推理 LLM 里程碑
- [[PedagogicalRL-Thinking]]: 以 DeepSeek-R1-Qwen3-8B 为基础模型训练教学推理

## 相关概念
- [[GRPO]]
- [[Thinking Reward]]
