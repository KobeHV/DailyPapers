---
type: concept
aliases: [Muon, Muon优化器]
---

# Muon Optimizer

## 定义
一种基于 Newton-Schulz 迭代的优化器，通过对梯度矩阵进行近似正交化来更新参数，在大模型训练中展现出比 AdamW 更快的收敛速度和更好的训练稳定性。

## 数学形式

Muon 更新（Nesterov 变体）：

$$\mathbf{U}_t = \text{HybridNewtonSchulz}\left(\mathbf{M}_t + \mu \mathbf{G}_t\right)$$

$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \cdot \mathbf{U}_t \cdot \max(\text{RMS}(\mathbf{U}_t), \epsilon)$$

Newton-Schulz 迭代：

$$\mathbf{X}_k = a \mathbf{X}_{k-1} + b (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top) \mathbf{X}_{k-1} + c (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1}$$

## 核心要点
1. 对矩阵参数进行整体正交化，而非逐元素更新
2. Hybrid Newton-Schulz: 前 8 步快速收敛系数 + 后 2 步精调系数
3. 与 ZeRO 结合需要特殊设计（knapsack 分配 + 混合策略）
4. 可与 BF16 Newton-Schulz + 随机量化 MoE 梯度来减半通信量

## 代表工作
- [[DeepSeek-V4]]: 对大部分模块使用 Muon，嵌入和 RMSNorm 保留 AdamW
- Jordan et al. (2024): Muon 原始提出
- Liu et al. (2025): Muon 在大模型训练中的可扩展性验证

## 相关概念
- [[Newton-Schulz Iterations]]
- [[AdamW]]
- [[ZeRO]]
