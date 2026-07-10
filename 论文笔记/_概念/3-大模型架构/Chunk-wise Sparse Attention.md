---
type: concept
aliases: [块级稀疏注意力, 块稀疏注意力]
---

# Chunk-wise Sparse Attention

## 定义
将输入序列划分为等大小的块（chunk），通过选择性关注部分块来实现线性或次二次复杂度的注意力机制。

## 核心要点
1. 序列被划分为大小为 $S$ 的非重叠块
2. 每个查询 token 仅关注局部滑动窗口 + 全局选择的 top-K 个远端块
3. 有效计算成本与序列长度近似呈线性关系
4. 块选择的质量直接影响模型性能

## 代表工作
- [[HiLS-Attention]]: 端到端可学习的层次化块稀疏注意力
- [[Native Sparse Attention|NSA]]: 硬件对齐的原生稀疏注意力
- [[DashAttention]]: 可微分自适应稀疏层次注意力
- [[InfLLM]]: 密集-稀疏可切换注意力

## 相关概念
- [[Hierarchical Softmax]]
- [[Landmark Token]]
- [[Block Sparse Attention]]
