---
type: concept
aliases: [miniF2F, miniF2F benchmark]
---

# miniF2F

## 定义
一个形式化数学证明基准，包含 488 道来自 IMO、AIME、AMC 等数学竞赛题的形式化版本，广泛用于评估自动化定理证明系统。

## 核心要点
1. 488 个问题，覆盖初等到高等数学竞赛
2. 支持多种形式化语言（Lean、Isabelle、Metamath 等）
3. 分为 valid（验证集）和 test（测试集）两个子集
4. 用于评估形式化证明搜索算法的端到端性能

## 代表工作
- [[TreeThink]]: 在 miniF2F 上评估形式化证明搜索能力
- [[DeepSeek-Prover]]: 在 miniF2F 上达到 SoTA 性能

## 相关概念
- [[MATH500]]
