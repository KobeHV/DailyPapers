---
type: concept
aliases: [BFS-Prover]
---

# BFS-Prover

## 定义
一种使用广度优先搜索进行形式化定理证明的方法，其核心特点是使用归一化长度的累积对数概率作为节点评分函数。

## 数学形式
$$
\text{score}(s) = \frac{\sum_{t=1}^{|s|} \log P(t_t | t_{<t}, c)}{|s|^\alpha}
$$

## 核心要点
1. 使用长度归一化防止长序列获得不公平的低分
2. $\alpha$ 控制长度惩罚的强度
3. TreeThink 将其评分函数实现为 `normalized_lengths_evaluator`

## 代表工作
- [[TreeThink]]: 实现了 BFS-Prover 风格的 normalized_lengths 评估器

## 相关概念
- [[广度优先树搜索|BFTS]]
- [[蒙特卡洛树搜索|MCTS]]
