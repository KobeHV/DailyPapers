---
type: concept
aliases: [MDP, 马尔可夫决策过程]
---

# Markov Decision Process

## 定义
描述序列决策问题的数学框架，由状态空间、动作空间、转移概率和奖励函数组成。

## 数学形式
$$(S, A, P, R, \gamma)$$

- $S$: 状态空间
- $A$: 动作空间
- $P(s'|s, a)$: 状态转移概率
- $R(s, a)$: 奖励函数
- $\gamma$: 折扣因子

## 核心要点
1. RL 的核心数学框架
2. 满足马尔可夫性质：下一状态仅取决于当前状态和动作
3. 在 LLM 对话中，状态 $s_t$ = 对话历史

## 代表工作
- [[GRPO]]
- [[PedagogicalRL-Thinking]]: 将 LLM 辅导对话建模为 MDP

## 相关概念
- [[GRPO]]
