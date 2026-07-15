---
type: concept
aliases: [pass@k, pass at k]
---

# Pass@K

## 定义
衡量模型在 $K$ 次独立采样中至少产生一个正确结果的概率的评估指标，广泛用于代码生成和数学推理评估。是 [[Test-Time Scaling]] 的标准评估指标。

## 数学形式

$$\mathrm{Pass@}K = 1 - \frac{\binom{N-C}{K}}{\binom{N}{K}}$$

其中 $N$ 为总采样数，$C$ 为正确样本数。

## 核心要点
1. 无偏估计：公式修正了从有限样本中估计 Pass@$K$ 的偏差
2. 适用于评估 TTS：能捕捉"低概率但存在正确解"的困难问题
3. 与 [[Average@N|Avg@N]] 互补：Avg@N 反映平均质量，Pass@$K$ 反映"能否找到正确解"
4. 大 $K$ 时 Pass@$K$ 逼近模型在该问题上的理论上界
5. $K=1$ 时退化为单次采样的 pass rate

## 代表工作
- [[TemperatureScaling]]: 使用 Pass@1,024 和 Pass@All 作为核心评估指标
- Codex (Chen et al., 2021): 首次提出 Pass@$K$ 用于代码生成评估
- [[MATH500]]: 使用 Pass@$K$ 评估数学推理

## 相关概念
- [[Average@N]]
- [[Test-Time Scaling]]
- [[Best-of-N]]
- [[MATH500]]
