---
tags: [training, rl, fine-tuning, reasoning]
aliases: [Rejection Fine-Tuning, Rejection Sampling Fine-Tuning, RFT]
created: 2026-07-09
---

# RFT (Rejection Fine-Tuning)

## 定义

Rejection Fine-Tuning (RFT)，也称 Rejection Sampling Fine-Tuning，是一种通过采样-过滤-训练循环来改进语言模型的后训练方法。

## 工作流程

1. **采样 (Sampling)**: 从当前模型对每个 prompt 采样 $K$ 个完整输出。
2. **过滤 (Rejection/Filtering)**: 根据验证器信号（如最终答案正确性、代码执行结果）丢弃错误输出，仅保留正确的输出 $\mathcal{D}_{\text{filtered}} = \{(x, y) | r(y|x) = 1\}$。
3. **微调 (Fine-Tuning)**: 在过滤后的正确样本上进行标准的监督微调（next-token prediction）。

## 目标函数

$$J_{\text{RFT}}(\theta) = \mathbb{E}_{(x, y^+) \sim \mathcal{D}_{\text{filtered}}}\left[-\log \pi_\theta(y^+ | x)\right]$$

## 特点

### 优点
- 实现简单，无需策略梯度或价值函数。
- 初始阶段提升明显，能快速 amplifiy 已有的成功模式。
- 不需要负样本，训练稳定。

### 局限
- **早期平台期 (Early Plateau)**: 改进在某个点后停滞，无法发现全新的推理策略。
- **缺乏对比信号**: 只能从正确样本学习，无法从错误中学习（不能区分"有效的组合"和"无效的捷径"）。
- **捷径扩散 (Shortcut Proliferation)**: 容易生成大量表面上正确但实际无效的捷径式推理步骤。
- **选择性不足**: 探索是随机的，没有机制将探索集中在有效、可复用的结构上。

## 与 RL (GRPO) 的关键区别

| 维度 | RFT | RL (GRPO) |
|------|-----|-----------|
| 信号来源 | 仅正确样本 | 正确 + 错误样本（对比信号） |
| 探索效率 | 高熵、无引导探索 | 低熵、选择性的结构化探索 |
| 能否发现组合策略 | 否（平台期） | 是（Phase 2） |
| 捷径行为 | 大量无效捷径 | 少量有效组合 |
| 算法复杂度 | 低（纯监督学习） | 中（策略梯度+KL正则化） |

## 相关论文

- [[RL Post-Training]]: 在可控环境中系统比较 RFT vs RL (GRPO)，证明差距在于选择性而非探索量。
- DeepSeekMath (Shao et al., 2024): 使用 GRPO 替代 RFT 实现持续改进。

## 适用场景

- 初始基线快速提升
- 模型已有较高的 pass@k 但 pass@1 较低时
- 作为 RL 训练前的 warm-up 阶段
