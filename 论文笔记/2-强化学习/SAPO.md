---
title: "Soft Adaptive Policy Optimization"
method_name: "SAPO"
authors: [Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin]
year: 2025
venue: arXiv
tags: [reinforcement-learning, policy-optimization, soft-clipping, moe, qwen]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2511.20347v1
created: 2026-07-14
---

# 论文笔记：Soft Adaptive Policy Optimization (SAPO)

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team, Alibaba Inc. |
| 日期 | December 2025 |
| 对比基线 | [[GSPO]], [[DeepSeekMath|GRPO]], GRPO-R2 |
| 链接 | [arXiv](https://arxiv.org/abs/2511.20347) |

---

## 一句话总结

> 用**软门控函数**替代 GSPO/GRPO 的硬裁剪，让梯度随偏离程度平滑衰减而非直接截断，解决 MoE 训练不稳定。

---

## 核心贡献

1. **软门控函数**: 基于 sigmoid 的连续软门控替代 min/max 硬裁剪
2. **非对称温度**: $\tau_{neg} > \tau_{pos}$ 应对负优势梯度扩散问题
3. **免 Routing Replay**: 不需要 GRPO-R2 的路由重放机制

---

## 方法详解

### SAPO 目标函数

$$\mathcal{J}_{\text{SAPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t f_{i,t}(r_{i,t}(\theta))\cdot\hat{A}_{i,t}\right]$$

### 软门控函数

$$f_{i,t}(x) = \sigma(\tau_{i,t} \cdot (x-1)) \cdot \frac{4}{\tau_{i,t}}$$

其中 $\tau_{i,t} = \begin{cases}\tau_{pos} & \hat{A} > 0 \\ \tau_{neg} & \hat{A} < 0\end{cases}$

### 梯度权重

$$\nabla\mathcal{J} = \mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t \underbrace{w_{i,t}(\theta)}_{\text{平滑权重}}\cdot r_{i,t}(\theta)\cdot\nabla\log\pi_\theta(y_{i,t})\cdot\hat{A}_i\right]$$

其中 $w_{i,t}(\theta) = 4p_{i,t}(1-p_{i,t})$，$p = \sigma(\tau(r-1))$

### 硬裁剪 vs 软门控

| 方面 | GSPO (硬裁剪) | SAPO (软门控) |
|------|:-----------:|:------------:|
| 序列中一个 token 偏离 | **全部 token 被抑制** | 只衰减该 token |
| 梯度形式 | min/max 硬边界 | sigmoid **平滑**衰减 |
| MoE 路由 | 需 Routing Replay | **不需要** |
| 训练早期崩溃 | 容易 | **稳定** |

### 非对称温度设计依据

负优势 $\hat{A}<0$ 时，梯度会扩散到大量无关 token（每个词典中的 token 被降低概率）。$\tau_{neg} > \tau_{pos}$ 让负优势梯度衰减更快。

---

## 关键结果

| 对比 | GSPO | GRPO-R2 | **SAPO** |
|------|:----:|:-------:|:--------:|
| 训练稳定性 | 早期崩溃 | 早期崩溃 | **稳定** |
| AIME25 等 | 基线 | 基线 | **最优** |
| Qwen3-VL | 基线 | 基线 | **持续增益** |
| MoE 特殊处理 | 不需 | 需 R2 | **不需** |

---

## 关联笔记

### 同系列（阿里 Qwen RL 算法演进）
- [[GSPO]]: 序列级 importance ratio
- [[StabilizingRL]]: 形式化框架（同团队）
- [[SAO]]: 清华团队，异步单 rollout
- [[DeepSeekMath|GRPO]]: 共同改进起点

---

## 速查卡片

> [!summary] SAPO
> - **核心**: 软门控替代硬裁剪
> - **方法**: $f(x) = \sigma(\tau(x-1)) \cdot 4/\tau$，$\tau_{neg} > \tau_{pos}$
> - **结果**: MoE 训练稳定，Qwen3-VL 持续提升
> - **影响**: 首次提出非对称温度处理负优势扩散问题

---

*笔记创建时间: 2026-07-14*
