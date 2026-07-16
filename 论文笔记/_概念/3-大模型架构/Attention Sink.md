---
type: concept
aliases: [注意力汇]
---

# Attention Sink

## 定义
一种注意力机制技巧，为每个 attention head 引入可学习的"汇"（sink）logit，使 query 可以选择不关注任何 token（总 attention score < 1），从而减少对无关 token 的强制注意力分配。

## 数学形式

$$\alpha_{m,t,i} = \frac{\exp(s_{m,t,i})}{\exp(s_{m,t,i}) + \exp(\sigma_m)}$$

其中 $\sigma_m$ 为第 $m$ 个 head 的可学习 sink logit，$\exp(\sigma_m)$ 加到分母中。

## 核心要点
1. 每个 attention head 独立调整总 attention score
2. 允许 head "选择不关注"，总 attention score 可以接近 0
3. 在长序列中特别有用：减少噪声 token 的干扰
4. DeepSeek-V4 中 CSA 和 HCA 的 core attention 均使用此技巧

## 代表工作
- OpenAI (2025): GPT-OSS 中使用
- Xiao et al. (2024): StreamingLLM 中的 attention sink
- [[DeepSeek-V4]]: CSA 和 HCA 中应用

## 相关概念
- [[Compressed Sparse Attention]]
- [[Heavily Compressed Attention]]
