---
type: concept
aliases: [PoT, 程序思维]
---

# Program-of-Thought (PoT)

## 定义
Program-of-Thought (PoT) 是一种数学推理方法，引导语言模型生成 Python 程序而非自然语言推理步骤，通过执行程序来获得最终答案。

## 核心要点
1. 使用 Python 代码（math, sympy 等库）进行数学计算
2. 将计算过程从推理中分离，减少计算错误
3. DeepSeekMath 的 SFT 数据中包含了 PoT 格式的解决方案

## 代表工作
- [[DeepSeekMath]]: 在 SFT 阶段使用 PoT 格式数据，并在工具辅助推理评估中测试

## 相关概念
- [[Chain-of-Thought]]
- [[Tool-Integrated Reasoning]]