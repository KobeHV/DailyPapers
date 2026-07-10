---
type: concept
aliases: [地标 Token, 块标记 Token]
---

# Landmark Token

## 定义
一种添加到每个块末尾的特殊 token，其隐藏状态被用作该块的压缩摘要表示，用于块级别的路由和选择。

## 数学形式
在 [[HiLS-Attention]] 中，landmark token 的查询向量 $\mathbf{q}'_c$ 用于构造块摘要键对 $(\mathbf{k}'_c, b'_c)$：

$$
\hat{s}_{i,c} = \frac{\mathbf{q}_i^\top \mathbf{k}'_c}{\sqrt{d}} + b'_c
$$

## 核心要点
1. 放置在每个块的最后位置
2. 其查询向量被用作该块的代理查询
3. 通过从 landmark token 的 key 提取信息来生成块摘要
4. 可端到端训练，无需额外监督信号

## 代表工作
- [[HiLS-Attention]]: 使用 landmark token 进行端到端块质量估计
- Random-Access Infinite Context Length for Transformers: 原始 landmark token 方法

## 相关概念
- [[Chunk-wise Sparse Attention]]
- [[Hierarchical Softmax]]
