---
tags: [concept, rl, self-play, game-theory]
created: 2026-07-09
---

# Self-Play (自我博弈)

## 定义

Self-Play 是一种训练范式，模型通过与自身的历史版本或变体进行博弈来改进。源自 AlphaGo/AlphaZero 等经典 RL 方法，近年来被应用于 LLM 的推理能力提升。

## LLM 中的 Self-Play 方法

| 方法 | 机制 | 局限 |
|------|------|------|
| SPIN (Chen et al., 2024) | 当前策略 vs 自身过去生成 | 单模型盲点循环 |
| Self-Rewarding LMs (Yuan et al., 2024) | 模型作为自身的裁判 | 自我评估偏差 |
| Absolute Zero (Zhao et al., 2025) | 单模型演化为 proposer + solver | 评分者与被评者共享盲点 |
| R-Zero (Huang et al., 2025) | 从零数据自我进化 | 同上 |
| **Agon (Beliaev, 2026)** | **两个不同模型**的竞争博弈 | 需要两个行为不同的模型 |

## Self-Play vs Cross-Model Play

Self-Play 的核心局限：评分者和被评者是**同一模型或其变体**，因此评分者与被评者共享导致错误的盲点 → 平台饱和。

Agon 保持 self-play 的竞争框架但用**两个不同的策略**替换了"self"，使得评分者成为真正的不同 agent：
- 不同初始化 → 初始盲点差异
- 角色轮换 + 不同更新流 → 分歧在训练中增长
- 竞争奖励 → 博弈压力驱动双方改进

## 与其他概念的关系

- 区别于 [[Cross-Model RL]]: self-play 是同一模型的变体，cross-model 是不同的模型
- 与 [[GRPO]] 结合：Agon 将竞争博弈嵌入 GRPO 的 group-relative advantage 框架
- 博弈论基础：纳什均衡、对抗训练、minimax 优化

## 主论文

- [[Agon]] (Beliaev, 2026) - 将 self-play 推广为 cross-model play
