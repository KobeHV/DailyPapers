---
type: concept
aliases: [LLM-as-a-Judge, LLM Judge, 大模型作为评判器]
---

# LLM-as-Judge

## 定义

使用大语言模型作为自动化评估器，对另一个模型的输出进行质量评分或比较判断的范式。

## 数学形式

给定 prompt $x$ 和模型回复 $\hat{y}$，LLM judge $f$ 输出评分：

$$
s = f(x, \hat{y}, \mathcal{C})
$$

其中 $\mathcal{C}$ 为可选的评分标准集合。

## 核心要点

1. 作为人工评估的低成本替代方案，广泛应用于 Chatbot Arena、AlpacaEval、MT-Bench 等基准
2. 可输出 Likert 评分（1-N 分）、pairwise 比较、或基于 checklist 的多维度判断
3. Judge 模型的规模、对齐程度和推理能力直接影响评分质量
4. 结构化评分标准（如 Rubric）可显著提升 judge 的一致性和准确性，尤其对小规模 judge
5. 风险：judge 自身偏见（长度偏好、位置偏见）、reward hacking、与人类偏好不完全对齐

## 代表工作

- [[RaR]] (Rubrics as Rewards): 用 rubric 引导 LLM judge，发现结构化标准可缩小 judge 规模差距
- MT-Bench / Chatbot Arena: 开创 LLM-as-judge 评估范式
- Prometheus 2: 开源专用评估模型
- J1: 用 RL 训练 LLM judge 的推理能力

## 相关概念
- [[Rubric-based Evaluation]]
- [[RLHF]]
- [[Reward Hacking]]
- [[Likert Scale]]
