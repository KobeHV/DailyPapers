---
type: concept
aliases: [工具集成推理, TIR]
---

# Tool-Integrated Reasoning

## 定义
Tool-Integrated Reasoning（工具集成推理）是一种结合自然语言推理和代码/工具使用的数学推理方法，模型先生成推理步骤，然后用 Python 等工具进行计算验证。

## 核心要点
1. 结合了 CoT 的自然语言推理和 PoT 的代码计算
2. DeepSeekMath 的 SFT 数据包含 CoT、PoT 和 Tool-Integrated 三种格式
3. 在工具辅助评估中，DeepSeekMath-Instruct 7B 在 MATH 上达到 57.4%

## 代表工作
- [[DeepSeekMath]]: 在 SFT 和评估中使用 Tool-Integrated Reasoning

## 相关概念
- [[Chain-of-Thought]]
- [[Program-of-Thought]]