---
type: concept
aliases: [HCA, 重度压缩注意力]
---

# Heavily Compressed Attention

## 定义
一种高效的注意力架构，对 KV Cache 进行更高压缩比（$l_h \gg l_c$）的压缩，但不使用稀疏注意力，而是保持稠密注意力。

## 数学形式

KV 压缩（无重叠）：

$$\mathbf{k}^{\text{Comp}}_j = \sum_{i=(j-1)l_h}^{jl_h-1} \gamma_i \mathbf{k}_i, \quad \gamma = \text{Softmax}(\mathbf{W}_\gamma \mathbf{x} + \mathbf{b})$$

稠密 MQA 注意力：

$$\mathbf{o}_{t,m} = \text{CoreAttn}(\mathbf{q}_{t,m}, \mathbf{K}^{\text{Comp}}, \mathbf{V}^{\text{Comp}})$$

## 核心要点
1. 压缩比 $l_h$（如 128）远大于 CSA 的 $l_c$（如 4）
2. 不进行重叠压缩，与 CSA 的区别
3. 同样使用共享 KV MQA 和分组输出投影
4. 适合对计算效率要求更高、可接受更大信息损失的场景

## 代表工作
- [[DeepSeek-V4]]: 与 CSA 交错使用，形成混合注意力架构

## 相关概念
- [[Compressed Sparse Attention]]
- [[Multi-Query Attention]]
- [[DeepSeek Sparse Attention]]
