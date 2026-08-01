---
type: concept
aliases: [GSM8K, Grade School Math 8K]
---

# GSM8K

## 定义
由 OpenAI 发布的 grade-school math 基准数据集，包含 8,500 道高质量小学数学文字题，要求多步算术推理。

## 核心要点
1. 8,792 道题目，标准划分为 7,473 训练 / 1,319 测试
2. 题目需 2-11 步推理，覆盖加减乘除四则运算
3. 每道题配有逐步自然语言解答，适用于 [[Process Reward|过程级监督]]
4. 是评估 [[Chain-of-Thought|CoT]] 推理和数学能力的标准基准
5. 步骤结构使其特别适合 [[RLVR]] 中的奖励粒度研究

## 代表工作
- [[Reward Granularity in RLVR]]: 在 GSM8K 上系统比较五种奖励条件的 RLVR 效果
- Cobbe et al. (2021): 原始论文，训练验证器解决数学文字题
- Samineni et al. (2025): 在同一数据集上分析 RLVR 推理痕迹的局部一致性

## 相关概念
- [[RLVR]]
- [[Chain-of-Thought]]
- [[Process Reward]]
- [[Outcome Reward]]
- [[GRPO]]
