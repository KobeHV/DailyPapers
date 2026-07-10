---
type: concept
aliases: [层次化 Softmax, 层次化归一化]
---

# Hierarchical Softmax

## 定义
将标准 softmax 归一化解构为多级层次结构，逐步分配概率质量而非一次性全部分配。

## 数学形式
在 [[HiLS-Attention]] 中，注意力权重分解为：

$$
w_{i,j} = \underbrace{\frac{\exp(s_{i,j})}{Z_{i,c(j)}}}_{\text{intra-chunk (块内)}} \times \underbrace{\frac{\hat{Z}_{i,c(j)}}{\hat{\mathcal{Z}}_i}}_{\text{inter-chunk (块间)}}
$$

## 核心要点
1. **块内归一化**: 在单个块内部分配注意力质量，衡量块内 token 的相对重要性
2. **块间归一化**: 在不同块之间分配注意力质量，衡量块的全局重要性
3. 层级的代理质量可通过 LM 损失端到端优化
4. 提供了稀疏注意力的自然分解方式

## 代表工作
- [[HiLS-Attention]]: 首次将层次化 softmax 用于原生稀疏注意力

## 相关概念
- [[Chunk-wise Sparse Attention]]
- [[Landmark Token]]
