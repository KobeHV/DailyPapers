---
title: "Stabilizing Reinforcement Learning with LLMs: Formulation and Practices"
method_name: "StabilizingRL"
authors: [Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, An Yang, Jingren Zhou, Junyang Lin]
year: 2025
venue: arXiv
tags: [reinforcement-learning, formulation, theory, moe, routing-replay, qwen, training-stability]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2512.01374v1
created: 2026-07-14
updated: 2026-07-14
aliases: [Stabilizing RL with LLMs, MiniRL, Routing Replay, R2, R3]
---

# 论文笔记：Stabilizing RL with LLMs — Formulation and Practices

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | Qwen Team, Alibaba Inc. |
| **作者** | Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li, Huiqiang Jiang, Junrong Lin, Yuqiong Liu, Hao Lin, Chencan Wu, Feng Hu, An Yang, Jingren Zhou, Junyang Lin |
| **发表** | arXiv:2512.01374, 2025年12月 |
| **计算量** | 数十万 GPU 小时 |
| **对比基线** | [[DeepSeekMath|GRPO]], CISPO |
| **链接** | [arXiv](https://arxiv.org/abs/2512.01374) |

---

## 一句话总结

> **首次为 LLM 强化学习建立严格形式化框架**—证明 token 级目标是序列级目标的一阶近似，将训练不稳定归因于**训练-推理不一致性**和**策略过时性**两个误差源，并提出 MiniRL 最小化基线 + Routing Replay。

---

## 核心贡献

1. **形式化理论**: 证明 token 级替代目标是序列级真实目标的一阶近似
2. **误差分解**: 将 IS 权重拆解为两个可控误差源
3. **MiniRL 算法**: 仅需 IS 校正 + 裁剪掩码的最小可工作基线
4. **Routing Replay (R2/R3)**: 专门解决 MoE 路由不一致问题

---

## 形式化框架

### 符号系统

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 训练引擎中的策略（待优化） |
| $\mu_{\theta_{\text{old}}}$ | 推理引擎中的 rollout 策略（FP8 精度） |
| $\pi_{\theta_{\text{old}}}$ | 训练引擎中计算的行为策略（BF16 精度） |
| $\mathcal{J}^{\text{seq}}(\theta)$ | **真实的**序列级奖励期望 |
| $\mathcal{J}^{\text{token}}(\theta)$ | **替代的** token 级优化目标 |
| $R(x,y)$ | 序列级奖励（只有序列末尾有） |

### 序列级目标（真实目标）

$$\mathcal{J}^{\text{seq}}(\theta) = \mathbb{E}_{x\sim\mathcal{D}, y\sim\mu_{\theta_{\text{old}}}(\cdot|x)}\left[\frac{\pi_\theta(y|x)}{\mu_{\theta_{\text{old}}}(y|x)} R(x,y)\right]$$

梯度：

$$\nabla_\theta \mathcal{J}^{\text{seq}}(\theta) = \mathbb{E}\left[\frac{\pi_\theta(y|x)}{\mu_{\theta_{\text{old}}}(y|x)} R(x,y) \sum_{t=1}^{|y|} \nabla_\theta \log \pi_\theta(y_t|x,y_{<t})\right]$$

**问题**: 序列似然比 $\frac{\pi_\theta(y|x)}{\mu_{\theta_{\text{old}}}(y|x)}$ 数值范围极大，方差极高，无法直接优化。

### Token 级替代目标（实践中使用的）

$$\mathcal{J}^{\text{token}}(\theta) = \mathbb{E}\left[\sum_{t=1}^{|y|} \text{sg}\left[\frac{\pi_\theta(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})}\right] R(x,y) \log \pi_\theta(y_t|x,y_{<t})\right]$$

梯度：

$$\nabla_\theta \mathcal{J}^{\text{token}}(\theta) = \mathbb{E}\left[\sum_{t=1}^{|y|} \frac{\pi_\theta(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})} R(x,y) \nabla_\theta \log \pi_\theta(y_t|x,y_{<t})\right]$$

### 核心定理

**当 $\pi_\theta = \mu_{\theta_{\text{old}}}$ 时**：

$$\nabla_\theta \mathcal{J}^{\text{seq}}(\theta) = \nabla_\theta \mathcal{J}^{\text{token}}(\theta)$$

> **Token 级目标是序列级目标的**一阶近似。当策略变化不大时，优化 token 级目标等价于优化序列级目标。

### 两个误差源

将 token 级 IS 权重分解为两个因子的乘积：

$$\frac{\pi_\theta(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})}}_{\text{误差源 1: 训练-推理不一致性}} \times \underbrace{\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})}}_{\text{误差源 2: 策略过时性}}$$

#### 误差源 1: 训练-推理不一致性

$\frac{\pi_{\theta_{\text{old}}}}{\mu_{\theta_{\text{old}}}} \neq 1$，因为训练和推理引擎的精度不同：
- 训练引擎: BF16 → 精度较高
- 推理引擎: FP8 → 精度较低

**在 MoE 模型中更严重**：不同精度下专家路由可能不同，导致路由不一致。

#### 误差源 2: 策略过时性

$\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} \neq 1$，因为：
- 大 batch 拆分为多个小批量 → 后续小批量的 $\pi_\theta$ 已经偏离 $\pi_{\theta_{\text{old}}}$
- 异步 RL 中：模型更新多次后 rollout 还在使用旧版本

### 实践含义

| 误差源 | 对应技术 | 缓解方法 |
|:------|---------|----------|
| 训练-推理不一致性 | IS 校正（必须） | $\frac{\pi_{\theta_{\text{old}}}}{\mu_{\theta_{\text{old}}}}$ 的数值校正 |
| 策略过时性 | 裁剪机制 | 约束 $r_t(\theta)$ 在安全范围内 |
| MoE 路由不一致性 | Routing Replay | R2: 重放旧路由 / R3: 重放推理路由 |

---

## MiniRL 算法

### 目标函数

$$\mathcal{J}_{\text{MiniRL}}(\theta) = \mathbb{E}\left[\sum_{t=1}^{|y|} M_t \cdot \text{sg}\left[\frac{\pi_\theta(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})}\right] \hat{A}(x,y) \log \pi_\theta(y_t|x,y_{<t})\right]$$

其中：

**组归一化优势**:
$$\hat{A}(x,y) = R(x,y) - \mathbb{E}_{y'\sim\mu_{\theta_{\text{old}}}(\cdot|x)}[R(x,y')]$$

**裁剪掩码** $M_t$:

$$M_t = \begin{cases}
0 & \text{if } \hat{A} > 0 \text{ and } r_t > 1 + \epsilon_{high} \\
0 & \text{if } \hat{A} < 0 \text{ and } r_t < 1 - \epsilon_{low} \\
1 & \text{otherwise}
\end{cases}$$

### MiniRL vs GRPO vs CISPO

| 方面 | GRPO | CISPO | **MiniRL** |
|:----|:----:|:-----:|:----------:|
| 训练-推理 IS 校正 | ❌ 无 | ❌ 无 | **✅ 有** |
| 长度归一化 | ✅ | ✅ | **❌ 无** |
| 裁剪方式 | min/clip | 裁剪 IS 权重 | **裁剪掩码** |
| KL 项 | 直接加入损失 | 直接加入损失 | 无（可加） |

> **为什么取消长度归一化**：长度归一化改变目标是梯度方向，破坏一阶近似。

---

## Routing Replay (R2/R3)

### MoE 的额外复杂性

MoE 的每个 token 路由到不同专家，IS 权重变为：

$$\frac{\pi_\theta(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})} = \frac{\pi_\theta(y_t|x,y_{<t}, e^\pi_t)}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t}, e^\mu_{\text{old},t})}$$

其中 $e^\pi_t$ 和 $e^\mu_{\text{old},t}$ 是不同引擎路由的专家。

### 两种 Routing Replay

**R2 (Vanilla Routing Replay)**:
在训练引擎中**重放旧策略的路由** $e^\pi_{\text{old},t}$（而不是当前路由 $e^\pi_t$）。

$$\frac{\pi_\theta^{\text{R2}}(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t}, e^\pi_{\text{old},t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t}, e^\mu_{\text{old},t})}}_{\text{固定路由}} \times \underbrace{\frac{\pi_\theta(y_t|x,y_{<t}, e^\pi_{\text{old},t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t}, e^\pi_{\text{old},t})}}_{\text{在固定路由上计算 ratio}}$$

- ✅ 缓解**策略过时性**
- ❌ 对**训练-推理不一致性**无帮助
- ✅ 第一个小批量不改变目标策略（偏差小）

**R3 (Rollout Routing Replay)**:
在训练引擎中重放**推理引擎的路由** $e^\mu_{\text{old},t}$。

$$\frac{\pi_\theta^{\text{R3}}(y_t|x,y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t})} = \frac{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t}, e^\mu_{\text{old},t})}{\mu_{\theta_{\text{old}}}(y_t|x,y_{<t}, e^\mu_{\text{old},t})} \times \frac{\pi_\theta(y_t|x,y_{<t}, e^\mu_{\text{old},t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t}, e^\mu_{\text{old},t})}$$

- ✅ 同时缓解**两者**
- ❌ 第一个小批量开始就改变目标策略（偏差大，但高 off-policy 时优势显现）

---

## 关键实验结果

### 实验设置

| 项目 | 配置 |
|:-----|------|
| 模型 | Qwen3-30B-A3B-Base (MoE), 冷启动微调版 |
| 任务 | 数学推理 (4096 题), 二进制奖励 |
| 推理精度 | FP8（故意放大不一致性） |
| 训练精度 | BF16 |
| 生成长度 | 32,768 tokens |
| 计算量 | 每步 5-6 GPU 小时，总数十万 GPU 小时 |

### On-policy 结果（无 off-policy 更新）

| 配置 | 结果 |
|:----|:-----|
| **MiniRL + IS 校正** | **最佳性能和稳定性** |
| + 长度归一化 | 性能次优（目标有偏） |
| - 训练-推理 IS 校正 | **快速崩溃**（熵急剧下降） |
| + R3 | 无增益（on-policy 时不需要） |

### Off-policy 结果（小批量更新）

Off-policy 程度控制：大 batch = N × 小批量（N 次梯度更新）

| 配置 | N=2 | N=4 | N=8 |
|:----|:---:|:---:|:---:|
| 无裁剪 + 无 Routing Replay | ❌ 崩溃 | ❌ 崩溃 | ❌ 崩溃 |
| MiniRL + **R2** + 裁剪 | **✅ 最佳** | ✅ | ✅ |
| MiniRL + **R3** + 裁剪 | ✅ | **✅ 最优** | **✅ 最优** |
| 仅 R2（无裁剪） | ❌ 不稳定 | ❌ 不稳定 | ❌ 不稳定 |

> **裁剪和 Routing Replay 都是 off-policy 训练必需的**，缺少任意一个都会导致崩溃。

### R2 vs R3 选择指南

| Off-policy 程度 | N=2 | N=4 | N≥8 |
|:--------------:|:---:|:---:|:---:|
| 推荐 | **R2** | R3 > R2 | **R3** |
| 理由 | R2 偏差小 | R3 更稳定 | R3 优势明显 |

### 冷启动初始化实验

三种冷启动数据来源：
1. Qwen3-Max-Thinking-Preview
2. DeepSeek-R1-0528
3. gpt-oss-120b

**关键发现**: 三种初始化在长时间稳定 RL 后达到**可比的最终性能**。

> 稳定训练比冷启动初始化重要得多。差异应归因于 RL 本身，而非初始化细节。

---

## 实用训练指南

### 稳定性检查清单

- [ ] **IS 校正**: 始终补偿训练-推理精度差异
- [ ] **取消长度归一化**: 它破坏一阶近似
- [ ] **On-policy**: 只需 IS 校正，无需裁剪/Routing Replay
- [ ] **Off-policy**: 裁剪 + Routing Replay 两者必需
- [ ] **低 off-policy (N=2)**: 用 R2
- [ ] **高 off-policy (N≥4)**: 用 R3
- [ ] **监控这些指标**:
  - 训练-推理 KL 散度（突升 = 不稳定前兆）
  - 策略熵（骤降 = 崩溃前兆）

---

## 批判性分析

### 优点
1. **理论严谨**: 首次为 LLM RL 提供了一阶近似的形式化证明
2. **可操作的指南**: 给出了明确的何时用 R2/R3 的规则
3. **计算量大但可信**: 数十万 GPU 小时的实验提供了可靠结论

### 局限性
1. **只验证 MoE 模型**: Qwen3-30B-A3B，对 Dense 模型的结论可能不同
2. **只验证数学任务**: 代码、通用对齐等任务未验证
3. **R2/R3 的工程成本**: 需要存储路由信息，增加内存和通信开销

---

## 关联笔记

### 同系列（阿里 Qwen 团队）
- [[GSPO]] — 序列级 IS ratio，另辟蹊径解决同一问题
- [[SAPO]] — 软门控，不需要 Routing Replay

### 理论分析
- [[DeepSeekMath|GRPO]] — MiniRL 的形式化分析对象
- [[PPO]] — 基础算法框架

---

## 速查卡片

> [!summary] Stabilizing RL with LLMs
> - **核心**: Token 级目标是序列级目标的**一阶近似**
> - **方法**: $\nabla\mathcal{J}^{\text{token}} = \nabla\mathcal{J}^{\text{seq}}$ 当 $\pi_\theta = \mu_{\theta_{\text{old}}}$
> - **误差分解**: 训练-推理不一致性 × 策略过时性
> - **MoE 方案**: R2/R3 Routing Replay
> - **结果**: 数十万 GPU 小时验证，给出实用稳定训练指南

---

*笔记创建时间: 2026-07-14 | 深度版*
