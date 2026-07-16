---
type: concept
aliases: [MTP, 多token预测]
---

# Multi-Token Prediction

## 定义
一种训练策略，模型在预测下一个 token 的同时也预测后续多个 token，通过增加预测深度来增强模型的表征学习能力。

## 数学形式

$$\mathcal{L}_{\text{MTP}} = \sum_{d=1}^{D} \lambda_d \cdot \mathcal{L}_{\text{CE}}(p_\theta(x_{t+d} \mid x_{<t}), x_{t+d})$$

其中 $D$ 为 MTP 深度，$\lambda_d$ 为各深度的损失权重。

## 核心要点
1. 使用额外的 MTP 模块（独立于主模型）进行多步预测
2. MTP 损失权重通常设为 0.3，学习率衰减时降至 0.1
3. 推理时可丢弃 MTP 模块，不影响推理效率
4. 在 DeepSeek-V2/V3/V4 中，MTP 深度设为 1

## 代表工作
- [[DeepSeek-V3]]: 首次在 DeepSeek 系列中引入 MTP
- [[DeepSeek-V4]]: 沿用 V3 的 MTP 配置，不做修改
- Gloeckle et al. (2024): MTP 的学术探索

## 相关概念
- [[Fill-in-Middle]]
- [[Language Modeling]]
