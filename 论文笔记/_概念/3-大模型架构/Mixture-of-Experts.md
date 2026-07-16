---
type: concept
aliases: [MoE, 混合专家]
---

# Mixture-of-Experts

## 定义
一种模型架构范式，将 FFN 层替换为多个"专家"子网络，通过路由机制为每个 token 选择激活部分专家，在参数量大幅增加的同时控制计算成本。

## 数学形式

$$\mathbf{y} = \sum_{i \in \mathcal{S}} g_i(\mathbf{x}) \cdot \text{Expert}_i(\mathbf{x})$$

其中 $\mathcal{S}$ 为 top-k 路由选中的专家集合，$g_i(\mathbf{x})$ 为路由权重（affinity score）。

## 核心要点
1. 总参数量大但每个 token 激活参数少（稀疏激活）
2. 路由策略包括：top-k softmax 路由、Hash 路由
3. 负载均衡是关键挑战：auxiliary-loss-free 策略、序列级平衡损失
4. 通信是主要瓶颈：需 Expert Parallelism (EP) + 通信计算重叠

## 代表工作
- [[DeepSeekMoE]]: 细粒度专家 + 共享专家的 MoE 设计
- Mixtral、GPT-4、DeepSeek-V2/V3/V4 均采用 MoE

## 相关概念
- [[DeepSeekMoE]]
- [[Expert Parallelism]]
- [[Hash Routing]]
- [[Auxiliary-Loss-Free Load Balancing]]
