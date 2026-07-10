---
type: concept
aliases: [张量核心, GPU Tensor Core]
---

# Tensor Core

## 定义
NVIDIA GPU 上的专用硬件单元，可在单个时钟周期内执行 $4 \times 4$ 矩阵乘加运算 ($D = A \times B + C$)，是现代深度学习加速的关键硬件。

## 核心要点
1. 高效执行矩阵乘法（GEMM）操作
2. 矩阵 tile 维度至少为 16 才能充分利用计算能力
3. 稀疏注意力的 kernel 设计需要确保矩阵形状与 Tensor Core 对齐
4. [[HiLS-Attention]] 通过查询打包策略将计算形状从 $(G,d) \times (d,S)$ 扩展为 $(M \times G,d) \times (d,S)$

## 代表工作
- [[HiLS-Attention]]: 针对 Tensor Core 优化的硬件高效 kernel 设计
- [[Native Sparse Attention|NSA]]: 依赖大 GQA 组大小的 kernel 设计

## 相关概念
- [[GQA]]
- [[Chunk-wise Sparse Attention]]
