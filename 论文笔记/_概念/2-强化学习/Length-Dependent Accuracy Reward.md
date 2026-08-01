---
type: concept
aliases: [长度依赖准确率奖励, 长度正则化奖励, length-regularized reward]
---

# Length-Dependent Accuracy Reward

## 定义
一种奖励塑形技术，在组内对正确响应按其相对长度动态调制奖励：短于组内均值的正确响应获得放大奖励，长于均值的被惩罚。用于鼓励推理模型生成简洁解答。

## 数学形式

标准化长度偏差：

$$z = \frac{|o| - \mu}{\sigma + \epsilon}$$

其中 $\mu$ 和 $\sigma$ 由组内正确响应子集计算。

奖励函数：

$$R_{\text{accuracy}}(o|q) = \exp(-\alpha z)$$

其中 $\alpha > 0$ 控制长度惩罚强度。

## 核心要点
1. 区别于静态长度目标（如 cosine reward 或 golden length），使用组内相对分布动态校准
2. $\mu$ 和 $\sigma$ 仅从正确响应计算，而非全部响应，确保奖励信号与正确性解耦
3. 指数衰减使奖励在 $[0, \infty)$ 范围，简洁正解获得 $>1$ 奖励
4. 显著减少推理输出长度（~25%），同时略微提升 Pass@1

## 代表工作
- [[GRPO-LEAD]]: Zhang & Zuo, 2025. 首次在 GRPO 框架中提出组内标准化长度奖励。
- L1: Aggarwal & Welleck, 2025. 通过 RL 控制推理模型思考长度的相关工作。

## 相关概念
- [[Group Relative Policy Optimization]]
- [[Reward Shaping]]
- [[Reward Hacking]]
- [[Advantage Normalization]]
