---
type: concept
aliases: [LLM-as-Judge, LLM评估器]
---

# LLM Judge

## 定义
使用 LLM（如 GPT-4o-mini）自动评估文本质量或行为规范的机制，作为 RL 训练中的非可微奖励信号源。

## 核心要点
1. 在教育 RL 中常用于：Leak Detection（答案泄露检测）、Helpfulness Assessment（有帮助性评估）、Thinking Quality Evaluation（思考质量评估）
2. 替代人工标注，实现大规模自动化评估
3. 存在评估偏差风险，可能与人类判断不一致

## 代表工作
- [[PedagogicalRL-Thinking]]: 使用三个 LLM Judge（Leak, Help, Think）
- [[PedagogicalRL]]: Leak Judge + Help Judge

## 相关概念
- [[Thinking Reward]]
- [[Composite Reward]]
