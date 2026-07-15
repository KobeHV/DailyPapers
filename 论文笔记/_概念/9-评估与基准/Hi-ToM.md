---
type: concept
aliases: [High-order Theory of Mind, 高阶心智理论]
---

# Hi-ToM

## 定义
High-order Theory of Mind（高阶心智理论）基准，用于评估 LLM 在复杂社会推理场景中对多层嵌套信念和意图的理解能力。包含 100 道题。

## 核心要点
1. 测社会推理而非数学/代码——评估 LLM 推理能力的另一个维度
2. 需要理解多层级递归信念（"A 知道 B 认为 C 想要..."）
3. 对人类也需要推理努力，高阶嵌套更容易出错
4. [[TemperatureScaling]] 中 Qwen3-8B 的 Pass@1,024 已达 93%，但小模型（1.7B）仅 37%
5. 多温度投票在 Hi-ToM 上计算节省最大（78.7%）——因为很多问题被判定为"简单"

## 代表工作
- [[TemperatureScaling]]: Hi-ToM 上多温度缩放提升 +2.5 到 +18.0 点，小模型收益最大

## 相关概念
- [[AIME]]
- [[LiveCodeBench]]
- [[Pass@K]]
