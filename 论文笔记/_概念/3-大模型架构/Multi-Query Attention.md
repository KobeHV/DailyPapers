---
type: concept
aliases: [MQA, 多查询注意力]
---

# Multi-Query Attention

## 定义
一种注意力机制的变体，所有 query head 共享同一组 key-value head，大幅减少 KV Cache 大小和注意力计算量。

## 数学形式

$$\mathbf{o}_m = \text{Attention}(\mathbf{Q}_m, \mathbf{K}_{\text{shared}}, \mathbf{V}_{\text{shared}})$$

其中 $\mathbf{K}_{\text{shared}}, \mathbf{V}_{\text{shared}}$ 为所有 head 共享的单组 KV。

## 核心要点
1. KV Cache 大小减少为 GQA/MHA 的 $1/n_h$
2. 在 CSA 和 HCA 中使用（Shared KV MQA）
3. 与分组输出投影结合可进一步降低计算量
4. 可能损失一定的注意力表达能力

## 代表工作
- Shazeer (2019): MQA 原始提出
- [[DeepSeek-V4]]: CSA 和 HCA 中均使用 Shared KV MQA

## 相关概念
- [[Grouped Query Attention]]
- [[Multi-Head Attention]]
- [[Compressed Sparse Attention]]
- [[Heavily Compressed Attention]]
