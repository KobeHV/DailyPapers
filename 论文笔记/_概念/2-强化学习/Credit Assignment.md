---
type: concept
aliases: [信用分配, Credit Assignment]
---

# Credit Assignment

## 定义
强化学习中确定每个动作/决策对最终奖励贡献程度的过程，决定了哪些行为应该被强化或抑制。

## 核心要点
1. 序列决策中，延迟奖励使得 credit assignment 具有挑战性
2. GRPO 使用广播规则（uniform credit assignment）：所有 token 共享序列级 advantage
3. TACO 提出 context-aware 的 token-level credit 校准

## 代表工作
- [[GRPO]]: 使用简单的广播 credit assignment
- [[TACO]]: 提出 context-aware 的 token-level credit 校准

## 相关概念
- [[GRPO]]
- [[PPO]]
- [[RLVR]]
- [[TACO]]
