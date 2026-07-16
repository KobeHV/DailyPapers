---
type: concept
aliases: [RoPE, 旋转位置编码]
---

# Rotary Positional Embedding

## 定义
一种通过旋转变换将位置信息编码到 attention query/key 中的位置编码方法，自然地引入了相对位置信息。

## 数学形式

$$\mathbf{q}_m^\top \mathbf{k}_n = (\mathbf{R}^m \mathbf{q})^\top (\mathbf{R}^n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}^{n-m} \mathbf{k}$$

其中 $\mathbf{R}^m$ 为 $m$ 位置的旋转矩阵。

## 核心要点
1. 通过旋转变换隐式编码相对位置
2. 具有远程衰减性质：距离越远 attention score 越低
3. 在 DeepSeek-V4 中，CSA/HCA 只对最后 64 维施加 RoPE（Partial RoPE）
4. 由于 KV 同时作为 key 和 value，需对 core attention 输出也施加 RoPE 抵消绝对位置

## 代表工作
- Su et al. (2024): RoPE 原始论文
- [[DeepSeek-V4]]: Partial RoPE + core attention 输出位置抵消

## 相关概念
- [[Sinusoidal Positional Encoding]]
- [[Compressed Sparse Attention]]
- [[Heavily Compressed Attention]]
