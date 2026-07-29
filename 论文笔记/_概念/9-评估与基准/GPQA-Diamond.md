---
type: concept
aliases: [GPQA Diamond, GPQA, Graduate-Level Google-Proof Q&A]
---

# GPQA-Diamond

## 定义

GPQA (Graduate-Level Google-Proof Q&A) 的高质量子集，包含研究生水平的科学多项选择题，由领域专家验证且难以通过搜索引擎直接找到答案。

## 核心要点

1. 题目覆盖物理、化学、生物等研究生水平科学领域
2. "Google-Proof" 意味着答案不能简单通过搜索获得，需要真正的专业推理
3. Diamond 子集经过严格的质量筛选，是 GPQA 中最可靠的子集
4. 多项选择题格式（4 选 1），评估方式为 exact match 或答案字母匹配
5. 在 [[RaR]] 中用于验证 rubric 训练的泛化能力——虽然训练时用 rubric 评分，但在 multiple-choice 评估上也有提升

## 代表工作

- GPQA 原始论文（Rein et al., 2023）
- [[RaR]]: 验证 rubric 训练的跨格式泛化能力（GPQA-Diamond 提升 7%）

## 相关概念
- [[HealthBench]]
- [[LLM-as-Judge]]
- [[Rubric-based Evaluation]]
