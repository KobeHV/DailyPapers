---
type: concept
aliases: [DSA, DeepSeek稀疏注意力]
---

# DeepSeek Sparse Attention

## 定义
DeepSeek 提出的稀疏注意力方案，通过索引器选择 top-k 压缩 KV entries 进行注意力计算，而非对所有 tokens 进行稠密注意力，大幅减少长序列注意力计算量。

## 核心要点
1. 先在 CSA 中压缩 KV cache 再进行稀疏选择
2. Lightning Indexer：低秩索引器快速评估 query-key 相关度
3. 每个 query 只关注 $k_s$ 个压缩 KV entries
4. DeepSeek-V4 相比 V3.2 使用更小的 attention top-k

## 代表工作
- [[DeepSeek-V3]]: DSA 首次应用
- [[DeepSeek-V4]]: CSA 中集成 DSA + KV 压缩

## 相关概念
- [[Compressed Sparse Attention]]
- [[Heavily Compressed Attention]]
- [[Multi-Query Attention]]
