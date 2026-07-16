---
type: concept
aliases: [哈希路由]
---

# Hash Routing

## 定义
一种 MoE 路由策略，根据输入 token ID 通过预定义的哈希函数确定目标专家，而非使用可学习路由网络，完全避免了路由计算和负载均衡问题。

## 数学形式

$$\text{ExpertIndex} = \text{Hash}(\text{token\_id}) \bmod N_{\text{experts}}$$

## 核心要点
1. 无需训练路由网络参数，消除路由计算开销
2. 天然负载均衡（好的哈希函数确保均匀分布）
3. 同类 token 总是路由到相同专家，促进专业化
4. 在 DeepSeek-V4 中用于前几层 MoE，替代 dense FFN

## 代表工作
- Roller et al. (2021): Hash Layers 原始提出
- [[DeepSeek-V4]]: 前 3 个 MoE 层使用 Hash routing

## 相关概念
- [[Mixture-of-Experts]]
- [[DeepSeekMoE]]
- [[Auxiliary-Loss-Free Load Balancing]]
