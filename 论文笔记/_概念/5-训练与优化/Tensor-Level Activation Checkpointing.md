---
type: concept
aliases: [Tensor-Level AC, 张量级激活检查点]
---

# Tensor-Level Activation Checkpointing

## 定义
一种细粒度的激活检查点（activation checkpointing / gradient checkpointing）方案，以单个张量而非整个模块为粒度决定保留或重计算，在内存节省和重计算开销之间取得更精细的平衡。

## 核心要点
1. 基于 TorchFX 追踪完整计算图
2. 开发者只需标注需要 checkpoint 的张量，框架自动生成重计算子图
3. 自动去重：共享存储的张量（如 reshape 的输入输出）不会重复重计算
4. 零额外开销：直接复用存储指针，无需 GPU 显存拷贝
5. DeepSeek-V4 的训练框架核心创新之一

## 代表工作
- [[DeepSeek-V4]]: 训练框架中实现张量级激活检查点

## 相关概念
- [[Activation Checkpointing]]
- [[Gradient Checkpointing]]
- [[Automatic Differentiation]]
