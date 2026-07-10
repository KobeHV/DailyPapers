---
tags: [concept, rl, multi-model, reasoning]
created: 2026-07-09
---

# Cross-Model RL (跨模型强化学习)

## 定义

跨模型强化学习（Cross-Model RL）是一种 RL 训练范式，在训练过程中使用**两个或多个不同的模型/策略相互交互**来产生训练信号，而非单一模型在自己的 rollout 上进行自我改进。Agon 是这一范式的代表性实例。

## 为什么需要跨模型

单模型 RL（如 GRPO + 自我改进）存在结构性天花板：

1. **Closed loop**: 策略在自身产生的信号上优化，强化导致错误的盲点
2. **Self-correction 不可靠**: 模型检查自己的工作继承了导致错误的偏见 (Huang et al., 2024)
3. **Self-play 饱和**: 无论多少轮 self-play，评分者与被评者共享相同的盲点

跨模型 RL 通过引入**不同的策略作为外部评分者**打破这个闭环：
- 不同的盲点 → 互补的错误检测
- 竞争压力 → 推理质量隐式评分
- 共同优化 → 评分者随被评者一起进步（不断升级的标准）

## 关键条件

1. **能力相当** (comparable strength): 否则博弈退化为蒸馏
2. **行为不同** (different blind spots): 否则失去跨模型的意义

## 相关方法

- [[Agon]]: 竞争性跨模型 RL + draft-and-challenge 协议
- [[Multiagent Finetuning]]: 通过不同数据流多样化多个模型，但目标为合作集成
- [[Prover-Verifier Games]]: 对抗性训练，但角色固定（一个生成，一个验证）

## 主论文

- [[Agon]] (Beliaev, 2026)
