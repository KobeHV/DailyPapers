---
type: concept
aliases: [AIME 2024, AIME 2025, AIME Benchmark]
---

# AIME

## 定义
American Invitational Mathematics Examination（美国数学邀请赛），面向高中生的高难度竞赛数学考试。AIME 2024 和 AIME 2025 各含 30 道题，被广泛应用于评估 LLM 的高级数学推理能力。

## 核心要点
1. 难度远高于 [[MATH500]]——需要多步推理和创造性思维
2. 每题答案为 000-999 的整数，便于自动判分
3. 是目前 LLM 数学推理评估最常用的高难度基准之一
4. [[TemperatureScaling]] 中观察到 Pass@1,024 在 AIME 上为 20%-73%（模型规模不同），远低于 MATH500 的 88%-97%
5. 多数 SOTA 推理模型（o1、DeepSeek-R1）均以 AIME 作为核心评测

## 代表工作
- [[TemperatureScaling]]: AIME 2024/2025 上多温度缩放提升 +3.3 到 +16.7 点
- [[GRPO]]: DeepSeekMath 在 AIME 上的 RL 训练

## 相关概念
- [[MATH500]]
- [[LiveCodeBench]]
- [[Pass@K]]
