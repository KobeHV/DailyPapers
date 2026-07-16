---
type: concept
aliases: [TileLang DSL]
---

# TileLang

## 定义
一种用于 GPU/NPU 内核开发的领域特定语言（DSL），平衡了开发效率和运行时性能，允许在同一代码库中快速原型和深度迭代优化。

## 核心要点
1. 用于 DeepSeek-V4 的融合内核开发，替代数百个细粒度 ATen 算子
2. Host Codegen：将 host-side 逻辑移出 Python 执行路径，减少调用开销
3. SMT-Solver-Assisted Formal Integer Analysis：集成 Z3 求解器进行整数表达式分析
4. 默认关闭 fast-math，优先保证数值精度
5. 支持 bitwise reproducibility 的对齐（与 CUDA hand-written 基准对齐）

## 代表工作
- Wang et al. (2026): TileLang 原始论文
- [[DeepSeek-V4]]: 用于开发 CSA/HCA/mHC 等融合内核

## 相关概念
- [[DeepGEMM]]
- [[Expert Parallelism]]
