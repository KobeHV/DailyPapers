---
type: concept
aliases: [mHC, 流形约束超连接]
---

# Manifold-Constrained Hyper-Connections

## 定义
对标准 Hyper-Connections (HC) 的改进，将残差映射矩阵约束到双随机矩阵流形（[[Birkhoff Polytope]]）上，增强深层网络信号传播的数值稳定性。

## 数学形式

约束条件（Birkhoff polytope）：

$$\mathcal{M} = \{ \mathbf{P} \in \mathbb{R}^{n \times n} \mid \mathbf{P}\mathbf{1} = \mathbf{1}, \mathbf{1}^\top \mathbf{P} = \mathbf{1}, \mathbf{P} \geq 0 \}$$

Sinkhorn-Knopp 投影：

$$\mathbf{P}^{(t)} = \mathcal{T}_c(\mathcal{T}_r(\mathbf{P}^{(t-1)}))$$

其中 $\mathcal{T}_r, \mathcal{T}_c$ 分别为行归一化和列归一化。

## 核心要点
1. 残差映射矩阵的谱范数 $\|\mathbf{P}\|_2 \leq 1$，保证非扩张性
2. 输入/输出映射通过 Sigmoid 函数约束为非负有界
3. 动态参数化：参数分解为输入依赖（动态）和输入无关（静态）分量
4. 相比标准 HC，训练稳定性大幅提升

## 代表工作
- [[DeepSeek-V4]]: 在所有 Transformer block 中使用 mHC 替代标准残差连接
- Xie et al. (2026): mHC 原始论文

## 相关概念
- [[Birkhoff Polytope]]
- [[Sinkhorn-Knopp Algorithm]]
- [[Residual Connection]]
