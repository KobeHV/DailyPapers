---
name: moe
description: Mixture of Experts — 混合专家模型
metadata:
  type: reference
---

# MoE (Mixture of Experts)

## 一句话
通过稀疏激活的专家子网络扩展模型参数量，保持计算成本近似不变。

## 在 RL 中的挑战
MoE 模型在 RL 训练中面临**路由不一致**问题：
- 训练引擎和推理引擎的路由不同 → 训练-推理不一致性放大
- 专家路由波动 → Token 级 importance ratio 方差增大

## 解决方案
| 方法 | 方案 |
|------|------|
| [[StabilizingRL]] | Routing Replay (R2/R3): 重放旧路由 |
| [[GSPO]] | 序列级 IS ratio，天然不依赖路由 |
| [[SAPO]] | 软门控处理路由波动 |

## 关联
- [[StabilizingRL]]: MoE RL 训练的深入分析
- [[GSPO]]: 免 Routing Replay 的序列级方案
