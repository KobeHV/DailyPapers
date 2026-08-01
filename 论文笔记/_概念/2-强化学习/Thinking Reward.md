---
type: concept
aliases: [思考奖励, Think Reward, Thinking Score]
---

# Thinking Reward

## 定义
在强化学习训练中，显式评估并奖励 LLM 内部推理轨迹（`<think>` 标签中的内容）的教学质量，而非仅评估可见输出的奖励机制。

## 数学形式
$$r_{\text{think}} \in [0, 1]$$

由 LLM Judge（如 GPT-4o-mini）评估三个维度：
- 推理的教学适当性
- 对学生理解的考量
- 元认知意识

## 核心要点
1. 将 RL 优化目标从"输出质量"扩展到"思维质量"
2. 需要与结构化的思维提示（如 Polya 框架）配合使用才能发挥最大效果
3. 仅评估教学推理质量，不涉及数学正确性本身

## 代表工作
- [[PedagogicalRL-Thinking]]: 首次提出 Thinking Reward 概念，用于训练 LLM 数学导师
- Special-R1: 将 Thinking Reward 扩展为 Persona-Aware Thinking Reward，适配特殊教育

## 相关概念
- [[GRPO]]
- [[Composite Reward]]
- [[PedagogicalRL]]
- [[LLM Judge]]
