---
type: concept
aliases: [信息熵, Shannon Entropy, Token Entropy]
---

# Entropy

## 定义
信息论中度量不确定性的核心概念。在 LLM 中，token 级熵 $H(p) = -\sum_i p_i \log p_i$ 反映模型在当前位置的预测不确定性——低熵表示模型"确信"，高熵表示模型"犹豫"。

## 数学形式

$$H(p) = -\sum_{i} p_i \log p_i$$

## 核心要点
1. 低熵 = 模型对某个 token 高度确信（通常对应正确的推理路径）
2. 高熵 = 模型在多个 token 间犹豫（可能是推理关键决策点或不正确的路径）
3. [[TemperatureScaling]] 发现：简单问题上正确/错误轨迹的熵分布明显分离，但困难问题上此信号失效
4. 温度 $T$ 直接影响熵——高温分散概率质量，提高熵
5. 可作为自适应温度选择或提前退出的潜在信号

## 代表工作
- [[TemperatureScaling]]: 分析温度-熵动态，揭示熵作为正确性信号在困难问题上的局限
- Semantic Uncertainty (Kuhn et al., 2023): 使用熵检测模型的不确定性

## 相关概念
- [[Temperature Sampling]]
- [[KL散度]]
- [[Test-Time Scaling]]
