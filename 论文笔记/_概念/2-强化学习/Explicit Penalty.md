---
type: concept
aliases: [显式惩罚, 错误惩罚, negative reward for incorrect answers]
---

# Explicit Penalty

## 定义
在 RL 训练中对模型的不正确输出施加固定负奖励（如 -1），替代传统的零奖励。通过将期望奖励与正确概率建立线性关系，在 $P(\text{correct}) < 0.5$ 时期望为负，从而抑制猜测行为。

## 数学形式

含显式惩罚的奖励函数（结合长度奖励）：

$$R_{\text{accuracy}}(o|q) = \begin{cases} \exp(-\alpha z), & \text{if } o \text{ is correct} \\ -1, & \text{if } o \text{ is incorrect} \end{cases}$$

期望奖励近似（忽略长度项时）：

$$\mathbb{E}[R] \approx 2P(\text{correct}) - 1$$

## 核心要点
1. 对比传统二元奖励（0/1），显式惩罚使期望奖励零点在 50% 正确率
2. 直接抑制猜测行为：模型学到"不确定时不应作答"
3. 在 GRPO-LEAD 消融中，加显式惩罚后 Pass@1 提升 2.6%-6.2%
4. 代价是推理长度略微增加（模型更谨慎），在低推理预算下略有性能损失

## 代表工作
- [[GRPO-LEAD]]: Zhang & Zuo, 2025. 在 GRPO 框架中引入 -1 显式惩罚。
- [[RLVR]]: 可验证奖励 RL 中的奖励设计讨论。

## 相关概念
- [[Group Relative Policy Optimization]]
- [[Length-Dependent Accuracy Reward]]
- [[Reward Shaping]]
- [[Reward Sparsity]]
