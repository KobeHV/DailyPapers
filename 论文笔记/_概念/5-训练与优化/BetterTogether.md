---
type: concept
aliases: [两者结合, Fine-Tuning and Prompt Optimization]
---

# BetterTogether

## 定义
BetterTogether 是一种将 [[提示优化]]（Prompt Optimization）与 RL 策略梯度方法（如 [[GRPO]]/[[mmGRPO]]）组合的框架，通过两阶段流水线实现互补收益：先优化提示模板，再微调模型权重。

## 核心流水线

```python
# Stage 1: 提示优化
program_po = MIPROv2(metric).compile(program, trainset)

# Stage 2: 权重优化
program_rl = GRPO(metric).compile(program_po, trainset)
```

## 设计动机
- **提示优化** 优化输入分布（找到更好的 prompt 模板和示例）
- **权重优化** 优化模型行为（通过 RL 让模型更擅长遵循优化后的指令）
- 两者互补：PO 为 RL 提供更好的初始分布，RL 使模型在 PO 找到的最优提示下表现更好

## 实验收益
在 [[mmGRPO]] 中，BetterTogether 相比未优化的 CoT 提升约 **11%**，相比仅 PO 提升约 **5%**，相比仅 mmGRPO 提升约 **3%**。

## 代表工作
- Soylu et al., 2024. Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together. arXiv:2407.10930. (首次提出)
- [[mmGRPO]]: Ziems et al., 2025. 将 BetterTogether 扩展到多模块 LM 程序和在线 RL.

## 相关概念
- [[提示优化]]: 第一阶段方法
- [[GRPO]]: 第二阶段方法
- [[DSPy]]: 实现框架
