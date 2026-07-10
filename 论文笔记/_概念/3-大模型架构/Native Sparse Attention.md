---
type: concept
aliases: [NSA, NATIVE SPARSE ATTENTION, 原生稀疏注意力]
---

# Native Sparse Attention

## 定义
一种硬件对齐的原生可训练稀疏注意力机制，通过设计专门的 GPU kernel 来支持动态稀疏注意力模式，同时保持端到端可训练性。

## 核心要点
1. 硬件-软件协同设计，kernel 针对 Tensor Core 优化
2. 每个查询 token 独立处理，计算形状为 $(G,d) \times (d,S)$
3. 需要 GQA 组大小 $G \geq 16$ 才能充分利用 Tensor Core
4. 在域内短上下文任务上接近全注意力性能

## 代表工作
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
- [[HiLS-Attention]]: 在 NSA kernel 设计基础上改进，支持查询打包和企业组大小灵活性

## 相关概念
- [[Chunk-wise Sparse Attention]]
- [[Tensor Core]]
- [[GQA]]
