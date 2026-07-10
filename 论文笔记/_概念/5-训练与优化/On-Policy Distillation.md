---
type: concept
aliases: [On-Policy Cross-Stage Distillation, 在线跨阶段蒸馏, 同策略知识蒸馏]
---

# On-Policy Distillation

## 定义
一种将知识蒸馏与策略梯度 RL 融合的训练方法：学生模型在自身生成的 on-policy 轨迹上，将冻结的教师模型的 log-prob 比率直接转化为 RL 目标中的 advantage 信号，实现前序训练能力的跨阶段保留。

## 数学形式

蒸馏 Advantage:

$$\hat{A}_{i,t}^{\text{distill}} = \text{sg}\!\left[ \log \frac{\pi_{\theta_{\text{teacher}}}^{\text{infer}}(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta}^{\text{train}}(y_{i,t} \mid x, y_{i,<t})} \right]$$

其中:
- $\text{sg}[\cdot]$: Stop Gradient (教师信号为常数，不反向传播)
- 正值: 学生行为与教师一致 → 正 reward
- 负值: 学生行为偏离教师 → 负 penalty

注入 GRPO/PPO 目标:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda \cdot \hat{A}^{\text{distill}}$$

## 核心要点
1. **On-Policy 采样**: 学生模型自己生成轨迹，教师仅对这些真实轨迹评分，解决 off-policy 蒸馏的 exposure bias
2. **统一训练引擎**: 蒸馏与 RL 使用同一 slime 框架，无需独立代码库
3. **多 Teacher 支持**: 前序阶段 (SFT, Reasoning RL, Agentic RL) 的 final checkpoint 均可作为 Teacher
4. **抗遗忘**: 在 General RL 阶段通过教师信号"锚定"前序阶段获得的能力
5. 与传统 KL 散度蒸馏不同，此处蒸馏信号作为 RL 目标的一部分，而非独立损失项

## 代表工作
- [[GLM-5]]: 首个规模化使用 On-Policy Cross-Stage Distillation 的多阶段 RL pipeline
- [[DeepSeek-R1]]: 使用 off-policy SFT 数据做阶段性蒸馏 (不同机制)

## 相关概念
- [[Knowledge Distillation]]: 经典的知识蒸馏 (off-policy, KL 散度)
- [[GRPO]]: 蒸馏 advantage 注入的目标算法
- [[Catastrophic Forgetting]]: 该方法要解决的核心问题
- [[SFT]]: SFT 阶段也可作为 Teacher
- [[Exposure Bias]]: On-Policy 采样要解决的偏差
