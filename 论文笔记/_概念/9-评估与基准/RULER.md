---
type: concept
aliases: [RULER 评测, 长上下文检索评测]
---

# RULER

## 定义
一个综合性长上下文合成评测基准，通过在上下文中插入特定针（needles）并提出相关问题，评估模型从长输入中检索任务相关信息的能力。

## 核心要点
1. **S-N (Single Needle)**: 单针检索，从长文本中查找一个目标信息
2. **MK-MQ (Multi-Key Multi-Query)**: 多键多查询，检索多个相关信息
3. **VT (Variable Tracking)**: 变量追踪，追踪变量值的变化，最困难的任务
4. 常用于评估模型的实际有效上下文长度
5. 在 [[HiLS-Attention]] 中，小模型先通过 RULER 风格的 NIAH 任务进行训练

## 代表工作
- RULER: What's the Real Context Size of Your Long-Context Language Models?
- [[HiLS-Attention]]: 在多个上下文长度上评估 RULER，显示 512 倍外推能力

## 相关概念
- [[LongBench]]
- [[Chunk-wise Sparse Attention]]
