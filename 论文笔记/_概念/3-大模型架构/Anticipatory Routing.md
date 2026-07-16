---
type: concept
aliases: [预判路由, 预期路由]
---

# Anticipatory Routing

## 定义
一种 MoE 训练稳定性技术，通过解耦骨干网络和路由网络的同步更新来打破路由引起的异常值恶性循环：当前步使用历史参数的路由决策，而非同步更新。

## 数学形式

第 $t$ 步：特征计算使用当前参数 $\boldsymbol{\theta}_t$，但路由索引使用历史参数 $\boldsymbol{\theta}_{t-\tau}$ 计算

## 核心要点
1. 路由网络和骨干网络的同步更新会加剧训练不稳定
2. 预先计算并缓存路由索引，实现约 20% 的额外 wall-time 开销
3. 通过自动检测机制动态启用：仅在 loss spike 时激活
4. 与 [[SwiGLU Clamping]] 组合使用效果更佳

## 代表工作
- [[DeepSeek-V4]]: 解决万亿参数 MoE 训练中的 loss spike 问题

## 相关概念
- [[Mixture-of-Experts]]
- [[SwiGLU Clamping]]
- [[DeepSeekMoE]]
