---
type: concept
aliases: []
---

# BigMath

## 定义
大规模数学问题数据集，涵盖算术、代数和应用题等多种类型，用于训练和评估 LLM 的数学推理和辅导能力。

## 核心要点
1. 包含 10,000 训练问题 + 500 评估问题
2. 训练前会过滤问题：仅保留 Student Simulator 解题率在 1%–60% 范围的题目
3. 是 PedagogicalRL-Thinking 等工作的核心训练数据

## 代表工作
- [[PedagogicalRL-Thinking]]: 使用 BigMath 训练 LLM 数学导师

## 相关概念
- [[WBEB]]
