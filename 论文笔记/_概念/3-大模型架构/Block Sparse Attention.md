---
type: concept
aliases: [Naive BSA, 朴素块稀疏注意力]
---

# Naive Block Sparse Attention

## 定义
一种理想化的块稀疏注意力方法，通过计算全注意力获取精确的块质量（chunk mass），选择质量最高的 top-K 块进行稀疏注意力计算。

## 数学形式
块质量定义为块内 token 指数化注意力分数的总和：

$$
Z_{i,c} = \sum_{j \in \mathcal{T}_c} \exp(s_{i,j})
$$

## 核心要点
1. 需要计算全注意力来获取精确块质量，因此无计算优势
2. 作为理论基准，衡量稀疏注意力方法的性能上限
3. 块选择是硬 top-K 选择（非可微）
4. 为 [[HiLS-Attention]] 提供理论起点

## 代表工作
- [[HiLS-Attention]]: 从 Naive BSA 出发，推导可学习的块质量代理

## 相关概念
- [[Chunk-wise Sparse Attention]]
- [[Chunk Mass]]
