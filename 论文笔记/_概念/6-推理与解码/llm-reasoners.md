---
type: concept
aliases: [LLM-Reasoners, InternLM-reasoners]
---

# llm-reasoners

## 定义
一个用于 LLM 高级推理的库，支持多种推理增强方法，包括 Tree-of-Thought、RAP-MCTS、Guided Decoding 等。

## 核心要点
1. 由 OpenLMLab 开发，支持 InternLM 系列模型
2. 支持 ToT、MCTS（RAP）、Guided Decoding 等推理策略
3. 与 TreeThink 目标类似但缺乏形式化证明验证支持

## 代表工作
- [[TreeThink]]: 与 llm-reasoners 对比，TreeThink 更侧重形式化证明和异步执行

## 相关概念
- [[Tree-of-Thought]]
- [[蒙特卡洛树搜索|MCTS]]
