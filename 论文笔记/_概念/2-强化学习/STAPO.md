---
type: concept
aliases: [STAPO]
---

# STAPO

## 定义
一种梯度感知的 token-level 优化稳定方法，通过抑制梯度幅值过大的 token 更新来增强 GRPO 训练稳定性。

## 核心要点
1. 识别产生过大梯度的 token 并 dampen 其更新
2. 关注训练稳定性而非 credit 分配
3. 与 TACO 的出发点不同但互补

## 代表工作
- [[TACO]]: 对比 baseline，TACO 在所有基准上超越 STAPO

## 相关概念
- [[GRPO]]
- [[RLVR]]
