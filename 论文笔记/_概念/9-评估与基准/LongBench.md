---
type: concept
aliases: [长上下文基准, LongBench 评测]
---

# LongBench

## 定义
一个双语、多任务的长上下文理解评测基准，涵盖多个类别的长文本任务（如文档问答、摘要、代码库理解等），按上下文长度分段评估模型性能。

## 核心要点
1. 覆盖多种任务类型：单文档 QA、多文档 QA、摘要、Few-shot 学习等
2. 按上下文长度分组报告性能（0-4K, 4-8K, 8-16K, 16-32K, >32K）
3. 评估模型在真实长文本任务上的综合能力
4. [[HiLS-Attention]] 在 7B 规模下的主要评估基准

## 代表工作
- LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding
- [[HiLS-Attention]]: 在 LongBench 上超越全注意力基线

## 相关概念
- [[RULER]]
