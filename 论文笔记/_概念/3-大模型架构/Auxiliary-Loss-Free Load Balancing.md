---
type: concept
aliases: [Auxiliary-Loss-Free, 无辅助损失负载均衡]
---

# Auxiliary-Loss-Free Load Balancing

## 定义
一种 MoE 负载均衡策略，不通过辅助损失项影响训练目标，而是通过动态调整路由 bias 来平衡专家负载。

## 核心要点
1. 对负载过重的专家降低 bias，对负载过轻的专家提高 bias
2. Bias 更新速度：$\gamma = 0.001$（DeepSeek-V4 配置）
3. 补充以微小序列级平衡损失（weight=0.0001）防止单序列极端不平衡
4. 避免了辅助损失对模型性能的负面影响

## 代表工作
- [[DeepSeek-V3]]: 首次提出该策略
- [[DeepSeek-V4]]: 继承使用，移除路由目标节点数量约束

## 相关概念
- [[Mixture-of-Experts]]
- [[DeepSeekMoE]]
- [[Hash Routing]]
