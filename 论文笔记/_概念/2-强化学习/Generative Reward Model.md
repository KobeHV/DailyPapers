---
type: concept
aliases: [GRM, 生成式奖励模型]
---

# Generative Reward Model

## 定义
一种利用生成式大模型本身作为奖励模型的范式，模型直接在自然语言中评估响应质量，而非输出标量分数，将评判能力与生成能力统一。

## 数学形式

传统标量 RM: $r = f_\phi(x, y)$
生成式 RM: $r = \text{Parse}(g_\theta(x, y, \text{rubric}))$

其中 $g_\theta$ 是从自然语言评估中提取分数的过程。

## 核心要点
1. 模型原生作为 GRM，无需单独训练标量奖励模型
2. 对 GRM 本身也施加 RL 优化（联合优化）
3. 深度融合推理能力到评估过程中，评分更鲁棒
4. 只需少量多样化人工标注即可获得优异表现

## 代表工作
- [[DeepSeek-V4]]: 后训练阶段完全使用 GRM 替代标量奖励模型

## 相关概念
- [[Reward Model]]
- [[Reinforcement Learning from Human Feedback]]
- [[Group Relative Policy Optimization]]
