---
type: concept
aliases: [Tail-Aware Credit CalibratiOn, TACO]
---

# TACO

## 定义
一种针对 GRPO 风格 RLVR 的 token-level credit 校准方法，通过尾风险评分软性抑制不可靠低概率 token 的正向 advantage，解决 Positive-Credit Contamination 问题。

## 核心要点
1. 利用 token 采样概率 $p_{i,t}$ 和局部熵 $H_{i,t}$ 估计尾风险
2. 尾风险评分: $r^{\text{tail}}_{i,t} = -\log p_{i,t} - H_{i,t} + \log \alpha$
3. 软性抑制权重: $w_{i,t} = 1 - \lambda(1 - \exp(-r^{\text{tail}}_{i,t}))$（$r^{\text{tail}}_{i,t} > 0$ 时）
4. 仅调制正向 advantage，负向保持不变
5. 极低额外计算开销（仅使用前向传播已有统计量）

## 代表工作
- [[TACO]] (本论文)

## 相关概念
- [[GRPO]]
- [[RLVR]]
- [[Positive-Credit Contamination]]
- [[Tail-Risk Score]]
- [[Credit Assignment]]
