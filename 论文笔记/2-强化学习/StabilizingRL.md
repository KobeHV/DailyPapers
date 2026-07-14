---
title: "Stabilizing Reinforcement Learning with LLMs: Formulation and Practices"
method_name: "StabilizingRL"
authors: [Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, An Yang, Jingren Zhou, Junyang Lin]
year: 2025
venue: arXiv
tags: [reinforcement-learning, formulation, theory, moe, routing-replay, qwen]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2512.01374v1
created: 2026-07-14
---

# 论文笔记：Stabilizing RL with LLMs — Formulation and Practices

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team, Alibaba Inc. |
| 日期 | December 2025 |
| 对比基线 | [[DeepSeekMath|GRPO]], CISPO |
| 链接 | [arXiv](https://arxiv.org/abs/2512.01374) |

---

## 一句话总结

> **首次为 LLM 强化学习提供严格形式化框架**，证明 token 级目标是序列级目标的一阶近似，需最小化训练-推理不一致性和策略过时性。

---

## 核心贡献

1. **形式化理论**: 证明 token 级目标是序列级目标的一阶近似（当 $\pi_\theta = \mu_{\theta_{\text{old}}}$ 时）
2. **两误差源分解**: 重要性采样权重 = 训练-推理不一致性 × 策略过时性
3. **MiniRL 算法**: 仅需 IS 校正 + 裁剪掩码的最小化基线
4. **Routing Replay (R2/R3)**: MoE 特有的路由重放机制

---

## 方法详解

### 形式化框架

**序列级目标**（真实目标）:
$$\mathcal{J}^{\text{seq}}(\theta) = \mathbb{E}\left[\frac{\pi_\theta(y|x)}{\mu_{\theta_{\text{old}}}(y|x)}R(x,y)\right]$$

**Token 级替代目标**（实践中使用）的梯度:
$$\nabla_\theta\mathcal{J}^{\text{token}}(\theta) = \mathbb{E}\left[\sum_t\frac{\pi_\theta(y_t)}{\mu_{\theta_{\text{old}}}(y_t)}R(x,y)\nabla_\theta\log\pi_\theta(y_t)\right]$$

**核心定理**: 当 $\pi_\theta = \mu_{\theta_{\text{old}}}$ 时，两者梯度相等 → Token 级目标是序列级目标的**一阶近似**

### 误差源分解

$$\frac{\pi_\theta(y_t)}{\mu_{\theta_{\text{old}}}(y_t)} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t)}{\mu_{\theta_{\text{old}}}(y_t)}}_{\text{训练-推理不一致性}} \times \underbrace{\frac{\pi_\theta(y_t)}{\pi_{\theta_{\text{old}}}(y_t)}}_{\text{策略过时性}}$$

1. **训练-推理不一致性**: 训练 (BF16) 和推理 (FP8) 引擎的数值差异
2. **策略过时性**: 旧策略采样 vs 新策略，小批量越多越严重

### MoE 特有: Routing Replay

| 方案 | 缓解 | 原理 |
|------|------|------|
| **R2** (Vanilla) | 策略过时性 | 重放训练引擎的旧路由 |
| **R3** (Rollout) | 两者都缓解 | 重放推理引擎的路由 |

### MiniRL 算法

$$\mathcal{J}_{\text{MiniRL}}(\theta) = \mathbb{E}\left[\sum_t M_t\cdot\text{sg}\left[\frac{\pi_\theta(y_t)}{\mu_{\theta_{\text{old}}}(y_t)}\right]\hat{A}(x,y)\log\pi_\theta(y_t)\right]$$

裁剪掩码 $M_t$: 正优势时 ratio > $1+\epsilon_{high}$ → $M_t=0$；负优势时 ratio < $1-\epsilon_{low}$ → $M_t=0$

---

## 关键发现（数十万 GPU 小时实验）

| 设置 | 必选项 |
|------|--------|
| On-policy | **只需 IS 校正** |
| Off-policy N=2 | IS + 裁剪 + **R2** |
| Off-policy N≥4 | IS + 裁剪 + **R3** |
| 无 IS 校正 | **快速崩溃** |
| 冷启动初始化 | 长时间训练后差异**消失** |

### 实用指南
1. **始终使用 IS 校正**（训练-推理不一致性）
2. **避免长度归一化**（破坏一阶近似）
3. 监控训练-推理 KL 散度和熵作为稳定性指标
4. 冷启动初始化在长时间稳定 RL 后差异消失

---

## 关联笔记

### 同系列
- [[GSPO]]: 序列级 importance ratio（同团队）
- [[SAPO]]: 软门控（同团队）
- [[DeepSeekMath|GRPO]]: 本文分析的核心目标

### 对比
- [[SAO]]: 异步 RL，独立解决方案

---

## 速查卡片

> [!summary] Stabilizing RL with LLMs
> - **核心**: Token 级目标是序列级目标的一阶近似
> - **方法**: 分解 IS 权重 → 训练-推理不一致性 × 策略过时性
> - **结果**: 数十万 GPU 小时验证，给出实用稳定训练指南
> - **影响**: LLM RL 领域首个严格形式化框架

---

*笔记创建时间: 2026-07-14*
