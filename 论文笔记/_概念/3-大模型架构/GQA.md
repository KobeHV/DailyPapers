---
type: concept
aliases: [Grouped-Query Attention, 分组查询注意力]
---

# GQA

## 定义
一种介于多头注意力（MHA）和多查询注意力（MQA）之间的注意力变体，多个查询头共享同一个 Key-Value 头组，以减少 KV 缓存大小和加速推理。

## 核心要点
1. 查询头被分为若干组，每组内的查询头共享相同的 KV 头
2. 相比 MHA 显著减少 KV 缓存，相比 MQA 保持更好的模型质量
3. 稀疏注意力中需要同一组查询头共享块选择集合以实现 kernel 效率
4. 现代 LLM 的标配设计（如 Llama、Qwen、Mistral 等）

## 代表工作
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- [[HiLS-Attention]]: 在 GQA 框架下设计了组级块选择策略

## 相关概念
- [[Tensor Core]]
- [[Chunk-wise Sparse Attention]]
