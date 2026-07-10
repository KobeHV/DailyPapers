---
type: concept
aliases: [Mixture of Experts, 混合专家模型]
---

# MoE (Mixture of Experts)

## 定义
一种将模型的前馈层 (FFN) 替换为多个独立的"专家"子网络，通过门控路由机制对每个 token 动态选择少量专家参与计算的稀疏模型架构，在增大总参数量的同时保持激活参数量和计算量相对稳定。

## 数学形式

门控路由 (Sigmoid Gating):

$$g(x) = \sigma(x W_{\text{gate}}) \in \mathbb{R}^{E}$$

Top-K 选择:

$$\mathcal{E}_{\text{active}} = \underset{e \in \{1,\dots,E\}}{\text{Top-k}}(g(x), k=8)$$

专家输出聚合:

$$y = \sum_{e \in \mathcal{E}_{\text{active}}} g_e(x) \cdot \text{FFN}_e(x) + \text{FFN}_{\text{shared}}(x)$$

## 核心要点
1. 总参数量可极大 (744B) 而每 token 仅激活约 5.9% (40B)，训练推理效费比高
2. GLM-5: 256 routed experts + 1 shared expert, Top-8 激活, Sigmoid 门控
3. 辅助损失 (load balancing / auxiliary loss) 用于防止路由塌缩
4. 层数减少以降低专家并行 (Expert Parallelism) 通信开销
5. GLM-5 前 3 层为 Dense, 后 77 层为 MoE

## 代表工作
- [[GLM-5]]: 744B/40B, 256 experts, Top-8
- [[DeepSeek-V3]]: 671B/37B, 256 experts, Top-8
- [[Mixtral 8x7B]]: 47B/13B, 8 experts, Top-2
- [[Qwen2.5-Max]]: 大规模 MoE

## 相关概念
- [[DSA (DeepSeek Sparse Attention)]]: 注意力稀疏 + 专家稀疏 叠加
- [[Expert Parallelism]]: MoE 训练中的分布式策略
- [[Load Balancing Loss]]: 防止专家路由塌缩的辅助损失
