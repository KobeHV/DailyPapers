---
type: concept
aliases: [CSA, 压缩稀疏注意力]
---

# Compressed Sparse Attention

## 定义
一种混合注意力架构，先沿序列维度压缩 KV Cache，再在压缩后的 KV entries 上执行稀疏注意力，大幅降低长序列推理的 FLOPs 和 KV Cache 大小。

## 数学形式

KV 压缩：每 $l_c$ 个 KV entries 压缩为一个压缩 entry $\mathbf{k}^{\text{Comp}}_j$：

$$\mathbf{k}^{\text{Comp}}_j = \sum_{i=(j-1)l_c}^{jl_c-1} \alpha_i \mathbf{k}_i + \sum_{i=(j-1)l_c}^{jl_c-1} \beta_i \mathbf{v}_i$$

索引评分：

$$s_{t,j} = \sum_{m=1}^{n_h} w_m \cdot \text{ReLU}\left( \mathbf{q}_{I,m} \cdot \mathbf{k}^{\text{IComp}}_j \right)$$

## 核心要点
1. CSA 可压缩到 $1/l_c$ 的序列长度
2. 结合 Lightning Indexer 进行 top-k 稀疏选择
3. 额外引入 sliding window attention 分支增强局部依赖
4. 共享 KV Multi-Query Attention 减少 KV head 数量

## 代表工作
- [[DeepSeek-V4]]: CSA + HCA 混合注意力实现百万 token 上下文

## 相关概念
- [[Heavily Compressed Attention]]
- [[DeepSeek Sparse Attention]]
- [[Multi-Query Attention]]
