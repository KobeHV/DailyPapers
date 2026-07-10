---
type: concept
aliases: [Rotary Position Embedding, 旋转位置编码]
---

# RoPE

## 定义
RoPE (Rotary Position Embedding) 是一种将位置信息编码到注意力机制中的方法，通过旋转矩阵对 Query 和 Key 向量进行变换，使得内积自然包含相对位置信息。

## 核心要点
1. 通过旋转矩阵编码位置信息，支持相对位置编码
2. 具有良好的外推能力（可以处理比训练时更长的序列）
3. DeepSeek LLM 架构使用的标准组件之一

## 相关概念
- [[Multi-Head Attention]]
- [[GQA]]