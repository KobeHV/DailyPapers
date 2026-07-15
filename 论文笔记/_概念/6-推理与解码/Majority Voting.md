---
type: concept
aliases: [多数投票, Self-Consistency, Majority Vote]
---

# Majority Voting

## 定义
一种 [[Test-Time Scaling]] 的聚合策略：从模型中独立采样多条推理轨迹，统计最终答案的出现频次，选择最频繁的答案作为输出。也称为 Self-Consistency (Wang et al., 2023)。

## 核心要点
1. 比 [[Best-of-N]] 更简单——无需外部奖励模型或验证器
2. 前提假设：正确答案在不同采样中更一致，错误答案更分散
3. 采样温度 $T$ 需足够大以保证多样性（通常 $T=0.7$）
4. [[TemperatureScaling]] 表明：单温度 Majority Voting 受限于该温度的可解问题子集
5. 可与多温度策略结合：先温度内投票，再跨温度投票

## 代表工作
- Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2023): 提出 Majority Voting for CoT
- [[TemperatureScaling]]: 提出 [[Multi-Temperature Voting]] 扩展 Majority Voting 到温度维度

## 相关概念
- [[Test-Time Scaling]]
- [[Best-of-N]]
- [[Multi-Temperature Voting]]
- [[CoT]]
