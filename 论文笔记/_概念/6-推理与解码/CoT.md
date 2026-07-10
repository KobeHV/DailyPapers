---
type: concept
aliases: [CoT, Chain-of-Thought, 思维链]
---

# Chain-of-Thought (CoT)

## 定义
Chain-of-Thought（思维链）是一种 prompting 方法，通过在 prompt 中提供逐步推理的示例，引导 LLM 在生成最终答案前先产生中间推理步骤。

## 数学形式
$$\text{CoT: } y \sim \pi_{\theta}(y|x, \mathcal{E}) \text{ where } \mathcal{E} = \{z_1 \to z_2 \to \dots \to a\}$$
$$z_i: \text{ intermediate reasoning step}$$
$$a: \text{ final answer}$$

## 核心要点
1. 通过中间推理步骤显式分解复杂问题
2. Zero-shot CoT 只需 prompt "Let's think step by step"
3. Self-Consistency 通过对多次 CoT 输出做多数投票提高鲁棒性
4. Tree-of-Thoughts (ToT) 从线性 CoT 扩展为树状搜索
5. CoT 在数学、逻辑、常识推理任务上效果显著

## 代表工作
- [[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models]]: 首次提出 CoT
- [[Self-Consistency Improves Chain of Thought Reasoning in Language Models]]: 多数投票增强
- [[Tree of Thoughts: Deliberate Problem Solving with Large Language Models]]: ToT 扩展
- [[CPP]]: CoT 的变体——命题具体化提示

## 相关概念
- [[CPP]]: 命题具体化提示，CoT 的改进
- [[Tree-of-Thoughts]]: 树状推理扩展
- [[Compositional Reasoning]]: 组合推理
- [[Self-Play]]: 通过自博弈增强推理
