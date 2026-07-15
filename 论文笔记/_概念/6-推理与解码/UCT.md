---
type: concept
aliases: [UCB1, Upper Confidence Bound, Upper Confidence Bound for Trees]
---

# UCT (UCB1)

## 定义
UCT（UCB applied to Trees）是 MCTS 中用于节点选择的策略函数，通过上置信界公式在探索与利用之间取得平衡。

## 数学形式
$$
UCB1 = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}
$$

**含义**: 选择 UCB1 值最大的子节点进行扩展

**符号说明**:
- $w_i$: 节点 $i$ 的累计奖励
- $n_i$: 节点 $i$ 的访问次数
- $N$: 父节点的总访问次数
- $c$: 探索权重参数

## 核心要点
1. 第一项 $\frac{w_i}{n_i}$ 代表利用（exploitation），倾向于高平均奖励的节点
2. 第二项 $c \sqrt{\frac{\ln N}{n_i}}$ 代表探索（exploration），倾向于访问次数少的节点
3. $c$ 越大探索越强，$c$ 越小利用越强

## 代表工作
- [[TreeThink]]: 使用 UCB1 作为 MCTS 的默认选择策略
- [[蒙特卡洛树搜索|MCTS]]: UCB1 是 MCTS 最常用的选择策略

## 相关概念
- [[蒙特卡洛树搜索|MCTS]]
