---
title: "Proximal Policy Optimization Algorithms"
method_name: "PPO"
authors: [John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov]
year: 2017
venue: arXiv
tags: [reinforcement-learning, policy-gradient, trust-region, clipping, rl-foundation]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/1707.06347v2
created: 2026-07-14
---

# 论文笔记：Proximal Policy Optimization Algorithms

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | OpenAI |
| 日期 | August 2017 |
| 对比基线 | [[TRPO]], A2C, ACER |
| 链接 | [arXiv](https://arxiv.org/abs/1707.06347) |

---

## 一句话总结

> 将 TRPO 的复杂二阶信赖域优化简化为**一阶裁剪机制**，成为深度 RL 的默认算法。

---

## 核心贡献

1. **PPO-Clip 目标函数**: 通过裁剪概率比实现信赖域约束，无需二阶计算
2. **PPO-Penalty 自适应 KL**: 提供自适应 KL 惩罚系数作为替代方案
3. **多 Epoch 小批量更新**: 每个数据样本支持多轮更新，大幅提升样本效率

---

## 问题背景

### 要解决的问题
策略梯度方法如何在不破坏策略的前提下进行**大且稳定**的更新？

### 现有方法的局限
- 标准策略梯度每个样本只做一次更新，样本效率低
- [[TRPO]] 使用二阶 KL 约束效果好，但实现复杂、计算昂贵、不兼容 dropout

### 本文的动机
找到一个**既简单又稳定**的方法，兼顾 TRPO 的可靠性和 SGD 的简便性。

---

## 方法详解

### PPO-Clip 目标函数

**概率比**（importance ratio）:
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

**裁剪后目标**:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**机制**:
- 正优势 $\hat{A}>0$: 鼓励提高 $r$，但不超过 $1+\epsilon$
- 负优势 $\hat{A}<0$: 鼓励降低 $r$，但不低于 $1-\epsilon$
- 取 min 保证是**下界估计**（悲观约束）

### PPO-Penalty 自适应 KL

$$L^{KL-PEN}(\theta) = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t - \beta\text{KL}[\pi_{\theta_{\text{old}}}, \pi_\theta]\right]$$

自适应调整 $\beta$:
- KL 太小 → $\beta \leftarrow \beta/2$
- KL 太大 → $\beta \leftarrow \beta \times 2$

### 完整目标函数

$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\right]$$

- $L_t^{VF} = (V_\theta(s_t) - V_t^{\text{targ}})^2$: 价值函数损失
- $S[\pi_\theta](s_t)$: 熵奖励项

---

## 关键结果

### MuJoCo 连续控制

| 算法 | 平均得分 |
|------|:--------:|
| 无裁剪/惩罚 | -0.39 |
| **PPO-Clip ε=0.2** | **0.82** |
| 自适应 KL d=0.01 | 0.74 |

### Atari 49 游戏胜场

| 算法 | A2C | ACER | **PPO** |
|:----:|:---:|:----:|:-------:|
| 胜场 | 1 | 18 | **30** |

---

## 对比

### PPO vs TRPO

| 方面 | TRPO | PPO |
|------|:----:|:---:|
| 更新约束 | 硬 KL 约束（二阶，共轭梯度） | **软裁剪**（一阶，SGD/Adam） |
| 实现复杂度 | 复杂 | **简单** |
| 架构兼容 | 不兼容 dropout/参数共享 | **完全兼容** |
| 每样本更新 | 1 次 | **K 个 epoch 小批量** |

---

## 批判性思考

### 优点
1. 实现极简，几乎成为 RL 的标准算法
2. 裁剪机制直观有效，超参数 ε=0.2 鲁棒
3. 兼容多种架构和辅助任务

### 局限性
1. 裁剪可能导致梯度信息丢失
2. 仍需要价值网络（增加内存）
3. 对 LLM RL 场景需要修改（如 GRPO）

---

## 关联笔记

### 后续工作
- [[DeepSeekMath|GRPO]]: 去掉价值网络的 PPO 变体
- [[StabilizingRL|Stabilizing RL]]: PPO 在 LLM 中的形式化分析

### 前置工作
- [[TRPO]]: PPO 优化的理论基础
- [[GAE|Generalized Advantage Estimation]]: PPO 使用的优势估计器

---

## 速查卡片

> [!summary] PPO
> - **核心**: 裁剪替代目标实现稳定策略更新
> - **方法**: $L^{CLIP} = \mathbb{E}[\min(r\hat{A}, \text{clip}(r,1\pm\epsilon)\hat{A})]$
> - **结果**: Atari 49 游戏中 30 胜，MuJoCo SOTA
> - **影响**: 至今仍是 RL 领域引用最高的算法之一 (>25000 引用)

---

*笔记创建时间: 2026-07-14*
