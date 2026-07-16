---
type: concept
aliases: [DualPipe, 双流水线]
---

# DualPipe

## 定义
DeepSeek 提出的双向流水线并行调度策略，通过 1F1B（one-forward-one-backward）交错调度，实现前向和反向计算的高效重叠，最小化流水线气泡。

## 核心要点
1. 在 DeepSeek-V3 中首次引入，V4 中继承并适配
2. 在 V4 中针对 mHC 增加的通信量和计算进行了调整
3. mHC 的 wall-time 开销被控制在 1F1B 流水线阶段的 6.7%
4. 与 ZeRO、Expert Parallelism 共同构成分布式训练栈

## 代表工作
- [[DeepSeek-V3]]: DualPipe 首次引入
- [[DeepSeek-V4]]: 适配 mHC 的 DualPipe 优化

## 相关概念
- [[Expert Parallelism]]
- [[ZeRO]]
- [[Manifold-Constrained Hyper-Connections]]
