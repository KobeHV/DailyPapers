---
type: concept
aliases: [DeepScaleR, DeepScaler dataset]
---

# DeepScaler

## 定义
一个大规模数学推理问题数据集，包含 AIME（1984-2023）、AMC（2023年以前）、Omni-MATH 和 Still 数据集的问题，总计约 40,300 个问题-答案对。每个问题有难度评级，广泛用于 GRPO 类 RL 训练。

## 数学形式

问题难度过滤：
- GRPO-LEAD: 仅使用难度 > 2.5 的问题（约 9,000 题）
- SFT 数据: 难度 > 1 的子集（约 13,000 题）用 QwQ-32B 生成解答

## 核心要点
1. 由 Luo et al., 2025 提出，是 DeepScaleR 模型的训练数据
2. 支持按难度分层，便于课程学习设计
3. 是多个 GRPO 改进工作（DAPO, GRPO-LEAD, Dr.GRPO）的标准训练集
4. 仅包含最终答案验证（outcome reward），无过程标注

## 代表工作
- DeepScaleR (Luo et al., 2025): 数据集初始提出者
- [[GRPO-LEAD]]: 基于 DeepScaler 进行难度过滤和 SFT 数据生成
- [[DAPO]]: DAPO-Math-17K 部分基于 DeepScaler

## 相关概念
- [[AIME]]
- [[Group Relative Policy Optimization]]
- [[Outcome Reward]]
