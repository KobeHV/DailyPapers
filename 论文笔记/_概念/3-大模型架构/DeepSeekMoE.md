---
type: concept
aliases: [DeepSeek混合专家]
---

# DeepSeekMoE

## 定义
DeepSeek 系列模型的 MoE 架构，采用细粒度路由专家 + 共享专家的设计，通过更细粒度的专家划分实现更高的专家专业化程度。

## 数学形式

$$\mathbf{y} = \sum_{i \in \mathcal{S}_r} g_i(\mathbf{x}) \cdot \text{RoutedExpert}_i(\mathbf{x}) + \sum_{j \in \mathcal{S}_s} \text{SharedExpert}_j(\mathbf{x})$$

Affinity score（DeepSeek-V4）: $\text{Sqrt}(\text{Softplus}(\mathbf{W}_a \mathbf{x}))$

## 核心要点
1. 细粒度路由专家：专家数量多但每个专家容量小
2. 共享专家：始终激活，捕获通用知识
3. 使用 Auxiliary-Loss-Free 负载均衡 + 序列级平衡损失
4. DeepSeek-V4 中移除了路由目标节点数量约束
5. 前几层使用 Hash routing 替代 dense FFN

## 代表工作
- [[DeepSeek-V2]]: 首次提出 DeepSeekMoE
- [[DeepSeek-V3]]: 引入 Auxiliary-Loss-Free 策略
- [[DeepSeek-V4]]: 引入 Hash routing 前几层 + Sqrt(Softplus) activation

## 相关概念
- [[Mixture-of-Experts]]
- [[Auxiliary-Loss-Free Load Balancing]]
- [[Hash Routing]]
- [[Expert Parallelism]]
