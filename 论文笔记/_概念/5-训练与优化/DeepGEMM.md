---
type: concept
aliases: [DeepGEMM库]
---

# DeepGEMM

## 定义
DeepSeek 开发的高性能 FP8 GEMM（通用矩阵乘法）内核库，支持 batch-invariant 和 deterministic 计算，是 DeepSeek-V4 训练栈的核心组件。

## 核心要点
1. 替代 cuBLAS 实现端到端 batch-invariant 矩阵乘法
2. 包含 MegaMoE：融合 EP 通信和 MoE 计算的大内核
3. 对超小 batch size 不使用 split-k，通过优化实现匹配或超越标准 split-k 性能
4. 开源地址：https://github.com/deepseek-ai/DeepGEMM

## 代表工作
- Zhao et al. (2025): DeepGEMM 发布
- [[DeepSeek-V4]]: 训练和推理中的核心矩阵乘法库

## 相关概念
- [[Expert Parallelism]]
- [[TileLang]]
- [[Matrix Multiplication]]
