---
type: concept
aliases: [avg@n, Pass@1, Average Accuracy]
---

# Average@N

## 定义
模型在 $N$ 次独立采样中的平均正确率，即 $\text{Avg@}N = C/N$（$C$ 为正确样本数）。等价于单次采样的期望正确率（Pass@1）。

## 数学形式

$$\mathrm{Avg@}N = \frac{C}{N}$$

## 核心要点
1. 反映模型的平均输出质量，而非"是否能找到正确解"
2. 可用于评估确定性行为（$T=0$），此时 Avg@1 = Pass@1
3. [[TemperatureScaling]] 指出其局限性：Avg@$N$ 无法区分"50% 时间正确"和"2% 时间正确但总能找到"的问题
4. 对于 [[Test-Time Scaling|TTS]] 评估，应优先使用 [[Pass@K]] 而非 Avg@$N$
5. 多个温度下的 Avg@$N$ 几乎一致——意味着均值无法揭示温度差异

## 代表工作
- [[TemperatureScaling]]: 系统分析了 Avg@$N$ 在 TTS 评估中的不足

## 相关概念
- [[Pass@K]]
- [[Test-Time Scaling]]
