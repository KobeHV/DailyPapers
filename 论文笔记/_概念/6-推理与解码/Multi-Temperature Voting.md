---
type: concept
aliases: [多温度投票, Two-Stage Voting, Multi-Temperature Voting Algorithm]
---

# Multi-Temperature Voting

## 定义
[[TemperatureScaling]] 论文提出的两阶段投票算法，用于降低多温度 [[Test-Time Scaling|TTS]] 的计算开销。先在每个温度内部投票，再跨温度投票聚合，对"简单"问题提前退出。

## 核心要点
1. **阶段 1 — 温度内投票**: 每个温度 $T_i$ 内，若最频繁答案频率 $\ge \tau_{intra}=0.8$，则该温度"自信"
2. **阶段 2 — 跨温度投票**: 所有温度都自信后，聚合各温度多数答案；若全一致（$\tau_{cross}=1.0$），标记为"简单问题"并退出
3. 不满足条件的问题使用全温度 Pass@All 策略（标记为"困难问题"）
4. 计算节省 26.9%-78.7%，简单数据集（Hi-ToM）节省最多
5. 配合温度子集裁剪（排除 $T=0.1-0.3$）可进一步压缩

## 代表工作
- [[TemperatureScaling]]: 提出该算法

## 相关概念
- [[Majority Voting]]
- [[Test-Time Scaling]]
- [[Temperature Sampling]]
- [[Pass@K]]
