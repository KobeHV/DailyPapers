---
type: concept
aliases: [EP, 专家并行]
---

# Expert Parallelism

## 定义
一种分布式训练/推理策略，将 MoE 层中的不同专家分布到不同设备上，通过跨设备通信完成 token 的分发（Dispatch）和结果收集（Combine）。

## 数学形式

Dispatch: $\mathbf{x}_i \to \text{Expert}_{r(i)}$（token $i$ 发送到其路由的专家所在设备）

Combine: $\sum_i \text{Expert}_{r(i)}(\mathbf{x}_i) \to \text{output}$（收集各专家输出）

## 核心要点
1. 通信（Dispatch/Combine）是主要瓶颈
2. DeepSeek-V4 的细粒度 EP：将专家分批调度，通信与计算 overlap
3. Pull-based dispatch 避免高通知延迟
4. 理论 speedup：DeepSeek-V4-Flash 配置下达 1.92x
5. 开源实现 MegaMoE（基于 CUDA，集成于 DeepGEMM）

## 代表工作
- [[DeepSeek-V4]]: 细粒度 EP 方案，单一大核融合通信计算

## 相关概念
- [[Mixture-of-Experts]]
- [[DeepGEMM]]
- [[DualPipe]]
