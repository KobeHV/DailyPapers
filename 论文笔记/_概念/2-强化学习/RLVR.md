---
tags: [rl, reasoning, llm, training]
aliases: [RL with Verifiable Rewards, 可验证奖励强化学习, Outcome Reward RL]
created: 2026-07-09
---

# RLVR (Reinforcement Learning with Verifiable Rewards)

## 定义

RLVR (Reinforcement Learning with Verifiable Rewards) 是一种利用**可自动验证的二元/标量奖励信号**进行强化学习后训练的范式。与依赖学习型奖励模型 (Reward Model, RM) 的 RLHF 不同，RLVR 使用确定性、无噪声的验证器信号。

## 核心要素

### 1. Verifiable Reward（可验证奖励）
奖励信号由确定性的验证函数产生：

$$r(y|x) = \begin{cases} 1 & \text{if output is correct} \\ 0 & \text{otherwise} \end{cases}$$

验证方式包括：
- **数学**: 最终答案数值匹配、Lean/Isabelle 形式化验证
- **代码**: 单元测试通过率、执行结果比对
- **逻辑推理**: 形式化验证器、约束求解器

### 2. Policy Optimization（策略优化）
使用策略梯度方法（而非监督学习）优化模型：
- **GRPO** (Group Relative Policy Optimization): 对每组采样进行归一化，利用组内对比信号。
- **PPO** (Proximal Policy Optimization): 使用学习型价值函数的经典方法。
- **REINFORCE** with baseline: 更简单的 Monte Carlo 策略梯度。

### 3. KL Regularization（KL 正则化）
通过 KL 散度惩罚项约束策略不要偏离参考模型太远：
$$J(\theta) = \mathbb{E}[r] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

## 关键算法：GRPO

### 流程
1. 对每个 prompt $x$ 采样 $G$ 个输出 $\{y_i\}$。
2. 用验证器计算每个输出的奖励 $r_i \in \{0, 1\}$。
3. **组归一化**: $\tilde{r}_i = \frac{r_i - \mu_G}{\sigma_G}$
4. 将归一化奖励作为优势 $\hat{A}_{i,t} = \tilde{r}_i$（对输出 $i$ 的所有 token 相同）。
5. 使用 PPO 风格的 clipped objective 更新策略，同时施加 KL 惩罚。

### 对比信号的本质
当奖励为二元时，组归一化后的优势自动产生对比信号：
- 正确样本 ($r=1$): 正优势 → 增加概率
- 错误样本 ($r=0$): 负优势 → 减少概率
- 相对幅度取决于组内正确/错误比例

## RLVR vs RLHF

| 维度 | RLVR | RLHF |
|------|------|------|
| 奖励来源 | 确定性验证器 | 学习型奖励模型 |
| 奖励噪声 | 无（精确） | 有（模型误差） |
| 适用任务 | 数学、代码、形式推理 | 开放式文本生成 |
| 奖励密度 | 通常是结果奖励 | 可以是结果或过程奖励 |
| Reward Hacking | 低（规则明确） | 高（可能利用RM漏洞） |
| 可扩展性 | 自动标注 | 需人工偏好数据 |

## RLVR vs RFT

见 [[RL Post-Training]] 和 [[RFT (Rejection Fine-Tuning)]] 中的详细对比。
核心差异：RLVR 利用对比信号（从错误中学习），RFT 仅从正确样本学习。

## 代表性工作

- **DeepSeek-R1** (DeepSeek, 2025): 大规模 RLVR 训练，用 GRPO + 规则验证器在数学和代码上取得突破。
- **DeepSeekMath** (Shao et al., 2024): 在数学领域比较 GRPO vs RFT，发现 GRPO 持续优于 RFT。
- **RL Post-Training Builds Compositional Reasoning Strategies** (Abdulsalam et al., 2026): 在可控环境中证明 RLVR 能组合原始技能。
- **STaR / Quiet-STaR** (Zelikman et al.): 自举式推理训练，利用正确推理链的自训练。
- **Open-Reasoner-Zero**: 开源复现 DeepSeek-R1 的 RLVR 训练。

## 开放问题

1. RLVR 的组合能力是否随模型规模涌现？
2. 过程奖励 (process reward) 是否能加速 Phase 1（原始技能强化）？
3. 如何在非可验证领域（创意写作、对话）扩展 RLVR？
4. 二元奖励的稀疏性在长链推理中是否构成瓶颈？
5. 离线 RLVR（仅使用预采样的数据训练）是否可行？

## 相关笔记

- [[GRPO]]: RLVR 的核心算法
- [[RL Post-Training]]: RLVR 组合推理的机制分析
- [[DeepSeek-R1]]: 真实世界 RLVR 案例
- [[RFT (Rejection Fine-Tuning)]]: 对比基线
- [[Verifiable Reward]]: 可验证奖励的设计原则
