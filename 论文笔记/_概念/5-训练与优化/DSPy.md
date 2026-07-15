---
type: concept
aliases: [DSPy, Declarative Self-improving Python]
---

# DSPy

## 定义
DSPy (Declarative Self-improving Python) 是一个用于优化 LM 程序的框架，提供声明式编程模型来定义、组合和自动优化多模块 LM 程序。由斯坦福 NLP 团队开发。

## 核心特性

1. **声明式模块**: 提供 `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct` 等预定义模块
2. **自动优化器**: 包括 `BootstrapFewShot`, `MIPROv2`, `GRPO` (mmGRPO) 等
3. **多模块支持**: 支持组合多个 LM 调用形成复杂程序管线

## 优化器家族

| 优化器 | 类型 | 说明 |
|--------|------|------|
| BootstrapFewShot | 少样本 | 自动生成示例 |
| MIPROv2 | 提示优化 | 贝叶斯提示搜索 |
| dspy.GRPO | RL 微调 | 基于 mmGRPO 的策略梯度优化 |

## 代表工作
- Khattab et al., 2023. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.
- [[mmGRPO]]: Ziems et al., 2025. 在 DSPy 中实现多模块 GRPO 优化器.

## 相关概念
- [[提示优化]]: DSPy 的优化器类别之一
- [[GRPO]]: DSPy 中的 RL 优化器
