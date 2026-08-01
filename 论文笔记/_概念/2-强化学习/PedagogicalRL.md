---
type: concept
aliases: [教学强化学习]
---

# PedagogicalRL

## 定义
将强化学习应用于训练 LLM 作为智能辅导系统（ITS）的方法，通过 Leak Detection 和 Helpfulness 两个维度的奖励信号优化教学行为。

## 数学形式
基于 GRPO 训练的 LLM 导师，奖励信号包含：
- Leak Judge: 检测是否直接泄露答案
- Help Judge: 评估回复对学生的帮助程度

## 核心要点
1. 由 Dinucu-Jianu et al. (2025) 首次提出
2. 仅评估可见输出的质量，不涉及思维过程
3. 是 PedagogicalRL-Thinking 的直接前置工作

## 代表工作
- Dinucu-Jianu et al. (2025): 首次提出 PedagogicalRL 范式
- [[PedagogicalRL-Thinking]]: 将 PedagogicalRL 扩展到思考阶段

## 相关概念
- [[GRPO]]
- [[Thinking Reward]]
- [[LLM Judge]]
- [[Intelligent Tutoring System]]
