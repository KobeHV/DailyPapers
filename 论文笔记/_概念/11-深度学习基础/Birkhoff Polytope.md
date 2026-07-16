---
type: concept
aliases: [双随机矩阵流形]
---

# Birkhoff Polytope

## 定义
所有 $n \times n$ 双随机矩阵（每行每列和均为 1 且所有元素非负）构成的凸多面体集合，是置换矩阵多面体的凸包。

## 数学形式

$$\mathcal{M} = \{ \mathbf{P} \in \mathbb{R}_{\geq 0}^{n \times n} \mid \mathbf{P}\mathbf{1} = \mathbf{1}, \mathbf{1}^\top \mathbf{P} = \mathbf{1} \}$$

性质：$\|\mathbf{P}\|_2 \leq 1$，且 $\mathcal{M}$ 在矩阵乘法下封闭。

## 核心要点
1. 谱范数上界为 1 保证非扩张性变换
2. 乘法封闭性保证深层堆叠时的稳定性
3. 在 mHC 中用作残差映射的约束空间
4. 通过 [[Sinkhorn-Knopp Algorithm]] 实现投影

## 代表工作
- [[DeepSeek-V4]]: mHC 将残差映射约束到 Birkhoff 多面体

## 相关概念
- [[Sinkhorn-Knopp Algorithm]]
- [[Manifold-Constrained Hyper-Connections]]
