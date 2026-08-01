---
type: concept
aliases: [复合奖励, 多组件奖励]
---

# Composite Reward

## 定义
在 RL 训练中将多个不同维度的奖励信号加权组合为一个综合奖励函数，以同时优化多个目标。

## 数学形式
$$r = r_{\text{sol}} + (r_{\text{ped}} - 1.0) \cdot \lambda_{\text{ped}} + (r_{\text{think}} - \theta) \cdot \lambda_{\text{think}}$$

- $r_{\text{sol}}$: 求解正确率
- $r_{\text{ped}}$: 教学质量（Leak + Help）
- $r_{\text{think}}$: 思考质量
- $\lambda_{\text{ped}} = 0.75$, $\lambda_{\text{think}} = 0.3$, $\theta = 0.6$

## 核心要点
1. 不同奖励维度可能相互竞争，需要精心设计权重
2. 使用阈值（$\theta$）可以实现低于阈值的负奖励
3. 复合奖励设计是 RLHF/RL 训练的核心工程挑战

## 代表工作
- [[PedagogicalRL-Thinking]]: 三部分复合奖励（解题 + 教学规范 + 思考质量）

## 相关概念
- [[Thinking Reward]]
- [[PedagogicalRL]]
- [[GRPO]]
