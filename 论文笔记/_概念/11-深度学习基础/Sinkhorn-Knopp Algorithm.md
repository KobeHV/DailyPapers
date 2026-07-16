---
type: concept
aliases: [SK算法, Sinkhorn迭代]
---

# Sinkhorn-Knopp Algorithm

## 定义
一种通过交替行列归一化将非负矩阵投影到双随机矩阵（[[Birkhoff Polytope]]）流形的迭代算法。

## 数学形式

初始化 $\mathbf{P}^{(0)} = \exp(\tilde{\mathbf{P}})$，然后迭代：

$$\mathbf{P}^{(t)} = \mathcal{T}_c(\mathcal{T}_r(\mathbf{P}^{(t-1)}))$$

其中 $\mathcal{T}_r(\mathbf{P})_{ij} = \mathbf{P}_{ij} / \sum_k \mathbf{P}_{ik}$（行归一化），$\mathcal{T}_c$ 类似（列归一化）。

在 [[DeepSeek-V4]] 中，迭代次数 $t_{\max} = 20$。

## 核心要点
1. 保证收敛到双随机矩阵（行和=1，列和=1）
2. 在 mHC 中用于约束残差映射矩阵
3. 计算效率高，适合 GPU 实现
4. 是连接矩阵和概率分布的桥梁

## 代表工作
- [[DeepSeek-V4]]: mHC 中使用 Sinkhorn-Knopp 约束残差映射

## 相关概念
- [[Birkhoff Polytope]]
- [[Manifold-Constrained Hyper-Connections]]
