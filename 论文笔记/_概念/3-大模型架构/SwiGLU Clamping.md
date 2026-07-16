---
type: concept
aliases: [SwiGLU Clamping, SwiGLU截断]
---

# SwiGLU Clamping

## 定义
对 SwiGLU 激活函数的线性分量和门控分量分别施加数值截断，防止训练过程中产生极端异常值导致 loss spike。

## 核心要点
1. 线性分量（$xW_1$）截断到 [-10, 10]
2. 门控分量（$\text{Sigmoid}(xW_2)$）上界截断到 10
3. 与 [[Anticipatory Routing]] 组合使用
4. 被实验证明有效消除异常值且不损害模型性能

## 代表工作
- OpenAI (2025): GPT-OSS 中使用
- [[DeepSeek-V4]]: 训练中全程使用

## 相关概念
- [[SwiGLU]]
- [[Anticipatory Routing]]
- [[Training Stability]]
