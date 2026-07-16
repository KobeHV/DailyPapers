---
type: concept
aliases: [Newton-Schulz迭代, NS迭代]
---

# Newton-Schulz Iterations

## 定义
一种通过迭代矩阵乘法逼近矩阵正交化的数值方法，将任意矩阵逐步变换为其奇异值分解中的正交因子 $\mathbf{U}\mathbf{V}^\top$。

## 数学形式

归一化初始化：$\mathbf{X}_0 = \mathbf{G} / \|\mathbf{G}\|_F$

Newton-Schulz 迭代：

$$\mathbf{X}_k = a \mathbf{X}_{k-1} + b (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top) \mathbf{X}_{k-1} + c (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1}$$

Hybrid Newton-Schulz（[[DeepSeek-V4]]）：
- 前 8 步：$(a,b,c) = (3.4445, -4.7750, 2.0315)$ 快速收敛
- 后 2 步：$(a,b,c) = (2, -1.5, 0.5)$ 精确稳定

## 核心要点
1. 目标是使矩阵的奇异值全部接近 1
2. 比完整 SVD 计算更高效，适合 GPU 上的批量执行
3. Hybrid 策略平衡了收敛速度和最终精度
4. 在 [[Muon Optimizer]] 中用于梯度矩阵的正交化

## 代表工作
- [[DeepSeek-V4]]: Muon 优化器中采用 Hybrid Newton-Schulz
- [[Muon Optimizer]]: Jordan et al. (2024), Liu et al. (2025)

## 相关概念
- [[Muon Optimizer]]
- [[Singular Value Decomposition]]
