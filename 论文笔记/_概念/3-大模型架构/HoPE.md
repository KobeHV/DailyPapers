---
type: concept
aliases: [混合位置编码, Hybrid Positional Encoding]
---

# HoPE

## 定义
一种混合位置编码方法，保留 [[RoPE]] 中旋转周期不超过预训练上下文长度的维度，将其余维度替换为 NoPE（无位置编码），同时提供长度外推和信息保留能力。

## 核心要点
1. 对 RoPE 维度进行筛选：仅保留旋转周期 $\leq$ 训练长度的维度
2. 超出训练长度的维度使用 NoPE（无偏置项）
3. 与 [[HiLS-Attention]] 配合使用时，可显著提升外推性能
4. 相比标准 RoPE，在稀疏注意力设置下困惑度更低

## 代表工作
- HoPE: a Novel Positional Encoding without Long-Term Decay (原始论文)
- [[HiLS-Attention]]: 采用 HoPE 替换 RoPE，实现 512 倍外推

## 相关概念
- [[RoPE]]
- [[Chunk-wise Sparse Attention]]
