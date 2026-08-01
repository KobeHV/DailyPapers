---
type: concept
aliases: [结果奖励, outcome-based supervision, outcome-based feedback, final-answer reward]
---

# Outcome Reward

## 定义
在 [[RLVR]] 中，仅评估最终答案正确性的奖励机制，提供稀疏的结果级监督信号。

## 数学形式
$$R_{\text{outcome}} = \begin{cases} 1 & \text{if } |\hat{a} - a| < \epsilon \\ 0 & \text{otherwise} \end{cases}$$

其中 $\hat{a}$ 为模型预测答案，$a$ 为参考答案，$\epsilon$ 为数值容差（通常为 $10^{-5}$）。

## 核心要点
1. 标签高效：仅需最终答案标签，无需步骤级标注
2. 允许模型自主探索推理路径，可能发现新颖解法
3. 推理过程不透明：模型可能通过捷径、记忆模式或表面线索达到正确答案
4. 小模型在稀疏反馈下容易产生"薄弱推理"（flaky reasoning）
5. 在 RLHF 和 RLVR 文献中被广泛使用

## 代表工作
- [[Reward Granularity in RLVR]]: 纯结果监督准确率 53.75%，低于纯过程监督近 10 个百分点；错误模式偏向推导和算术错误
- Wen et al. (2025): 展示纯最终答案奖励也能影响推理痕迹
- Tang et al. (2025): 通过 Jensen 下界缩放可验证奖励

## 相关概念
- [[Process Reward]]
- [[Reward Granularity]]
- [[RLVR]]
- [[RLHF]]
- [[Chain-of-Thought]]
