---
type: concept
aliases: [Likert评分, Likert量表, Likert Score, 李克特量表]
---

# Likert Scale

## 定义

一种常用的评分量表，要求评估者对某个陈述的同意程度或质量进行分级评分，通常为 1-5 或 1-10 的整数分。

## 核心要点

1. 在 LLM 评估中，[[LLM-as-Judge]] 通常使用 1-10 Likert 量表对回复质量打分
2. Direct-Likert: 仅给 prompt + 回复，直接打分，简单但一致性差
3. Reference-Likert: 额外提供参考答案作为对比基准
4. Rubric-Likert: 提供结构化评分标准引导打分（如 [[RaR]] 的 Implicit Aggregation）
5. 局限性：单一分数丢失了多维质量信息；不同 judge 之间的分数不可直接比较（需归一化）；易受 judge 偏见影响

## 代表工作

- MT-Bench (Zheng et al., 2023): 率先使用 GPT-4 1-10 Likert 评分
- [[RaR]]: 对比 Direct-Likert、Reference-Likert、Rubric-Likert 三种评分方式的效果

## 相关概念
- [[LLM-as-Judge]]
- [[Rubric-based Evaluation]]
- [[RaR]]
