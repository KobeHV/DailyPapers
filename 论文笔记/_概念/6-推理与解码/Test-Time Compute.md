---
type: concept
aliases: [推理时计算, Inference-Time Compute, Test-Time Compute Budget]
---

# Test-Time Compute

## 定义
在模型推理阶段可分配的计算资源总量，通常以 FLOPs、生成 token 数或采样轨迹数来衡量。是 [[Test-Time Scaling]] 的核心资源维度。

## 数学形式

$$\text{Compute Budget} = \sum_{i=1}^{m} K_i \cdot L_i$$

其中 $m$ 为温度数，$K_i$ 为每个温度的采样数，$L_i$ 为平均轨迹长度。

## 核心要点
1. 传统 TTS 将所有计算预算分配在单一温度下的 $K$ 维度
2. Temperature Scaling 将预算重新分配：部分给 $K$ 维度，部分给温度 $T$ 维度
3. 存在计算效率权衡——更多温度 = 更广覆盖，但 $K$ 减少 = 每温度探索深度降低
4. 两阶段投票算法通过提前退出来节省计算预算

## 代表工作
- [[TemperatureScaling]]: 提出将计算预算从 K 维度扩展到温度 T 维度
- Scaling Laws for Inference (Snell et al., 2024): 研究推理时计算的 scaling law

## 相关概念
- [[Test-Time Scaling]]
- [[Pass@K]]
- [[Temperature Sampling]]
- [[Multi-Temperature Voting]]
