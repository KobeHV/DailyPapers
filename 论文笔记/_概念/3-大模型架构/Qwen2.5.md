---
type: concept
aliases: [Qwen2.5, Qwen 2.5, 通义千问2.5]
---

# Qwen2.5

## 定义

阿里巴巴通义千问团队的 2.5 代开源大语言模型系列，覆盖 0.5B 至 72B 参数规模，在推理、数学、代码等多项基准上表现优异。

## 核心要点

1. 基于 Qwen2 架构改进，支持 128K 上下文窗口
2. 提供 Base 和 Instruct 版本，Instruct 版本经过 SFT + RLHF 对齐
3. 7B 版本是 LLM 后训练研究中广泛使用的 base policy（如 [[RaR]]、[[GRPO]] 相关工作）
4. 在 GRPO on-policy 训练中表现出良好的可训练性

## 代表工作

- Qwen2.5 技术报告（Qwen Team, 2024）
- [[RaR]]: 使用 Qwen2.5-7B 作为 base policy，用 GRPO + rubric 奖励训练
- [[GRPO]] / DAPO / SAPO 等多项 RL 工作中均使用 Qwen2.5-7B

## 相关概念
- [[GRPO]]
- [[RLVR]]
- [[LoRA]]
- [[SFT]]
