---
type: concept
aliases: [Rubric Evaluation, Checklist-based Evaluation, 评分量表评估, 评分标准评估]
---

# Rubric-based Evaluation

## 定义

使用结构化的、多项目的评分量表（Rubric）对模型输出进行细粒度评估的方法。每条评分标准检查响应的特定维度，最终汇总为整体评分。

## 核心要点

1. 评分标准通常包含多个类别：Essential（必要）、Important（重要）、Optional（可选）、Pitfall（常见错误）
2. 每条标准是自包含的、二元的检查项，不需要外部知识即可判断
3. 相比直接 Likert 评分，rubric 提供更细粒度、更一致、更可解释的评估
4. 可用于评估阶段（[[HealthBench]]），也可转化为训练奖励信号（[[RaR]]）
5. 关键设计考量：专家信号的重要性、标准粒度、权重分配、合成 vs 人工撰写

## 代表工作

- [[HealthBench]]: 48k 条医师撰写的医学评分标准
- [[RaR]]: 首次将评分量表系统化用于 on-policy RL 训练
- CPT (Gallego, 2025): 用评分标准生成 DPO 偏好对
- QA-Lign (Dineen et al., 2025): 宪法式 QA 分解用于对齐

## 相关概念
- [[LLM-as-Judge]]
- [[Likert Scale]]
- [[RaR]]
- [[HealthBench]]
