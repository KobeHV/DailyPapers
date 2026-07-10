---
type: concept
aliases: [投机解码, 推测解码]
---

# Speculative Decoding

## 定义
一种加速自回归解码的技术，使用一个轻量级草案模型（draft model）快速生成多个候选 token，然后由目标模型在一次前向传播中并行验证，在不改变输出分布的前提下提升推理速度。

## 核心要点
1. 核心思路：用快速草稿模型预测多个 token，大模型并行验证
2. 接受率决定了加速比
3. 与稀疏注意力可协同工作（相邻查询 token 打包）
4. 在 [[HiLS-Attention]] 中，打包的查询 token 可直接用于投机解码的场景

## 相关概念
- [[Tensor Core]]
- [[Chunk-wise Sparse Attention]]
