---
type: concept
aliases: [FP4 QAT, FP4量化感知训练]
---

# FP4 Quantization-Aware Training

## 定义
一种在训练过程中引入 FP4 (MXFP4) 量化模拟的量化感知训练方法，使模型在训练时就适应推理时的低精度量化，减小精度损失。

## 数学形式

前向传播：$\mathbf{W}_{\text{FP8}} = \text{Dequantize}(\text{Quantize}_{\text{FP4}}(\mathbf{W}_{\text{FP32}}))$

反向传播：$\nabla_{\mathbf{W}_{\text{FP32}}} = \nabla_{\mathbf{W}_{\text{FP8}}}$（Straight-Through Estimator）

## 核心要点
1. 在 [[DeepSeek-V4]] 的后训练阶段引入
2. 对 MoE 专家权重和 CSA indexer QK 路径进行 FP4 量化
3. FP4-to-FP8 反量化是无损的（FP8 E4M3 有更大的动态范围）
4. 推理时直接使用原生 FP4 权重，无需模拟量化
5. 可完全复用现有 FP8 训练框架

## 代表工作
- [[DeepSeek-V4]]: 后训练阶段 QAT，实现 MoE FP4 权重 + CSA Indexer FP4 加速

## 相关概念
- [[Quantization]]
- [[Mixture-of-Experts]]
- [[Compressed Sparse Attention]]
