---
type: concept
aliases: [BoN, Best-of-N Sampling, Rejection Sampling]
---

# Best-of-N

## 定义
最简单的 [[Test-Time Scaling]] 方法：从模型中独立采样 $N$ 个输出，使用奖励模型或验证器（verifier）选择最佳结果。本质上是一种推理时的拒绝采样。

## 核心要点
1. 无需训练，仅需一个打分函数（reward model / verifier / self-consistency）
2. 性能随 $N$ 提升，但存在 [[TemperatureScaling]] 中揭示的饱和现象
3. 与 [[Majority Voting]] 的区别：BoN 用外部打分器，Majority Voting 用答案频次
4. 计算开销与 $N$ 线性增长
5. 采样温度 $T$ 的选择对 BoN 效果至关重要——但此前被普遍忽视

## 代表工作
- [[TemperatureScaling]]: 揭示 BoN 在单温度下存在性能饱和，多温度可互补
- Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021): 使用 verifier 做 Best-of-N

## 相关概念
- [[Test-Time Scaling]]
- [[Pass@K]]
- [[Majority Voting]]
- [[Temperature Sampling]]
