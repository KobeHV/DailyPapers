---
type: concept
aliases: [DeepSeek Sparse Attention]
---

# DSA (DeepSeek Sparse Attention)

## 定义
一种基于动态内容感知的 token 级稀疏注意力机制，通过轻量级 Indexer 打分 + Top-k 选择，将传统密集注意力的 O(L²) 计算复杂度降至 O(L·k)，在大幅降低计算/显存开销的同时保持长上下文性能近乎无损。

## 数学形式

Indexer 打分与筛选:

$$s_i = q \cdot W_i + b_i, \quad g_i = \max(0, s_i), \quad \mathcal{I}_{\text{top-k}} = \underset{i}{\text{Top-k}}(g_i, k=2048)$$

稀疏注意力:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K_{\mathcal{I}_{\text{top-k}}}^T}{\sqrt{d_k}}\right) V_{\mathcal{I}_{\text{top-k}}}$$

## 核心要点
1. 两阶段流水线：Lightning Indexer (FP8 精度的 ReLU 门控快速粗筛) → Top-k Selector (仅对 k=2048 个最相关 token 执行完整 MLA 注意力)
2. Indexer 直接操作 MLA 压缩后的低秩潜向量 (rank=512)，每 FLOP 成本远低于主注意力
3. 通过继续预训练 (continue pre-training) 引入，仅需 ~20B tokens 即可完成适配 (vs 从头训练)
4. RL 训练中需使用确定性 top-k 算子 (torch.topk) 并冻结 Indexer 参数以保证稳定性
5. 相邻层 Indexer 输出 70-100% 重叠 → IndexCache 可进一步移除 75% Indexer 计算

## 代表工作
- [[GLM-5]]: 在 744B MoE 模型上应用 DSA，KV Cache -75%, 推理速度 +1.5-2x, 长文本性能损失 < 0.5%
- [[DeepSeek-V3.2]]: DSA 的原始提出者，使用 943B tokens 进行 DSA 适配
- [[IndexCache]]: 跨层复用 Indexer 输出，prefill 1.82x / decode 1.48x 加速

## 相关概念
- [[MLA (Multi-Head Latent Attention)]]: Indexer 在 MLA 压缩向量上操作
- [[FlashAttention]]: 替代的注意力加速方案 (IO-aware kernel fusion)
- [[Sliding Window Attention]]: 另一种稀疏注意力 (固定窗口 vs 动态选择)
