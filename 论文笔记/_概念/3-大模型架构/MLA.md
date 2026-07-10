---
type: concept
aliases: [Multi-Head Latent Attention, 多头潜注意力]
---

# MLA (Multi-Head Latent Attention)

## 定义
一种将注意力机制中的 Key 和 Value 投影到低秩潜空间 (latent space) 并进行压缩缓存的技术，在推理时动态重建完整 K、V 矩阵，从而大幅减少 KV Cache 显存占用。

## 数学形式

KV 压缩:

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c + d_r}$$

其中 $d_c = 512$ (压缩维度), $d_r = 64$ (RoPE 位置编码维度)

重建:

$$k_t^C = W^{UK} c_t^{KV}, \quad v_t^C = W^{UV} c_t^{KV}$$

## 核心要点
1. KV 仅缓存 ~576 维压缩向量 (512 + 64 RoPE)，相比 GQA 的 2048 维大幅缩减
2. Query 同样进行低秩压缩 (rank 2048) 后上投影
3. Muon Split: 将上投影矩阵按注意力头独立拆分后分别正交化，使 MLA 性能匹敌 GQA-8
4. MLA-256 变体: 头维度 192→256, 头数减 1/3, 解码计算量显著下降
5. 最初由 DeepSeek-V2 提出，GLM-5/DeepSeek-V3 系列广泛采用

## 代表工作
- [[DeepSeek-V2]]: MLA 首次提出
- [[DeepSeek-V3]]: MLA + MoE 架构的 671B 模型
- [[GLM-5]]: Muon Split 使 MLA 性能匹敌 GQA-8; MLA-256 降低解码计算量

## 相关概念
- [[DSA (DeepSeek Sparse Attention)]]: MLA + DSA 可叠加使用
- [[GQA (Grouped-Query Attention)]]: MLA 替代的目标注意力机制
- [[MQA (Multi-Query Attention)]]: 更激进的 KV 压缩方案 (单 KV head)
