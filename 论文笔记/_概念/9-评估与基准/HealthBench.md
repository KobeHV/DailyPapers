---
type: concept
aliases: [HealthBench, 健康基准]
---

# HealthBench

## 定义

由 5,000 个临床对话场景组成的医学 LLM 评估基准，使用 48,000+ 条医师撰写的细粒度评分标准（Rubric）来评估模型的安全性和有用性。

## 核心要点

1. 覆盖真实临床对话场景，而非简单的多项选择题
2. 评估五个核心维度：Communication quality（沟通质量）、Accuracy（准确性）、Completeness（完整性）、Instruction following（指令遵循）、Context awareness（上下文意识）
3. 评分标准由执业医师撰写，涵盖 Essential/Important/Optional/Pitfall 四个类别
4. 使用 GPT-4 作为 judge 进行自动化评估
5. [[RaR]] 论文引入 HealthBench-1k 子集用于消融实验

## 代表工作

- Arora et al. (2025): HealthBench 原始论文
- [[RaR]]: 将 HealthBench 作为 rubric-based 评估的主要 benchmark，同时用其训练数据合成评分量表

## 相关概念
- [[Rubric-based Evaluation]]
- [[LLM-as-Judge]]
- [[GPQA-Diamond]]
