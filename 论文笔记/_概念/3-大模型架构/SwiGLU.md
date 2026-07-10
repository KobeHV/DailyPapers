---
type: concept
aliases: [Swish-Gated Linear Unit]
---

# SwiGLU

## 定义
SwiGLU 是一种激活函数，结合了 Swish 激活函数和门控线性单元 (GLU)。它通过门控机制选择性地传递信息，已被证明在语言模型中优于标准 ReLU 或 GELU。

## 数学形式
$$
\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \odot (xV + c)
$$

## 核心要点
1. 结合 Swish 激活和门控线性单元
2. 在语言模型中通常比 ReLU/GELU 表现更好
3. DeepSeek LLM 架构使用的标准组件

## 相关概念
- [[GQA]]
- [[DeepSeek LLM]]