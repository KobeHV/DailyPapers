---
type: concept
aliases: []
---

# MetaMath

## 定义
MetaMath 是一种通过数据增强提升数学推理能力的方法，对 GSM8K 和 MATH 训练集进行改写和增强，然后在增强数据上微调 Llama-2 模型。

## 核心要点
1. 通过对数学问题进行多种形式改写进行数据增强
2. 在 DeepSeekMath 中被作为对比基线（MetaMath 70B 在 MATH 上 26.6%）
3. DeepSeekMath-Instruct 7B (46.8%) 大幅超越 MetaMath 70B (26.6%)

## 相关概念
- [[WizardMath]]
- [[DeepSeekMath]]