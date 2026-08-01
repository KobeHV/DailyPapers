---
type: concept
aliases: [QwQ, Qwen-QwQ]
---

# QwQ-32B

## 定义
Qwen 系列的一个 32B 参数推理增强模型，专门设计用于生成高质量分步数学推理过程。在 GRPO-LEAD 中作为 SFT 数据生成的 teacher 模型。

## 核心要点
1. 擅长生成结构化的分步数学解答
2. 在 GRPO-LEAD 中用于为 DeepScaler 的 ~13k 题生成 SFT 训练数据
3. 生成的推理过程质量显著影响下游 SFT→RL 的效果
4. 是 Qwen 系列的推理特化变体

## 代表工作
- [[GRPO-LEAD]]: 使用 QwQ-32B 生成 SFT 数据
- Qwen2.5-Math (Yang et al., 2024): Qwen 数学系列

## 相关概念
- [[DeepSeek-R1]]
- [[SFT]]
- [[DeepScaler]]
