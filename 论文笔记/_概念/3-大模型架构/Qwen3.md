---
type: concept
aliases: [Qwen3, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B]
---

# Qwen3

## 定义
阿里巴巴通义千问团队发布的开源大语言模型系列，支持 0.6B 到 8B 多个规模。[[TemperatureScaling]] 论文选择 Qwen3 系列作为主要实验模型，验证了多温度缩放在不同规模模型上的一致性收益。

## 核心要点
1. 支持多语言（中英文为主）和长上下文
2. 基座模型（非 instruct/chat 版本）具有较强的原始推理能力
3. [[TemperatureScaling]] 覆盖 0.6B/1.7B/4B/8B 四个规模，验证方法的跨规模泛化
4. 开源可用，被广泛用于推理研究
5. 推理性能随规模单调提升（AIME 2025 Pass@1,024: 0.6B 20% → 8B 60%）

## 代表工作
- [[TemperatureScaling]]: 在多温度 TTS 中验证 Qwen3 全系列
- Polaris-4B-Preview: 基于 Qwen3-4B 的 RL 训练推理模型

## 相关概念
- [[Test-Time Scaling]]
- [[RLHF]]
- [[AIME]]
