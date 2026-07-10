---
type: concept
aliases: [MHA, 多头注意力]
---

# Multi-Head Attention

## 定义
Multi-Head Attention (MHA) 是 Transformer 架构的核心组件，将输入投影到多个注意力头中并行计算注意力，然后将所有头的输出拼接并线性变换。

## 数学形式
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$
其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

## 核心要点
1. 每个注意力头学习不同的表示子空间
2. 是 Transformer 系列模型的标准组件
3. DeepSeekMath 使用 MHA 作为注意力机制基础

## 相关概念
- [[GQA]]: MHA 的变体，减少 KV 头数量
- [[RoPE]]: 旋转位置编码，与 MHA 配合使用