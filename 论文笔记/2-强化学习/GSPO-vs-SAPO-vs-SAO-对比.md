---
title: "GSPO vs SAPO vs SAO — 深度对比分析"
method_name: "GSPO-SAPO-SAO-Comparison"
authors: []
year: 2026
venue: 笔记
tags: [comparison, reinforcement-learning, grpo-variants, analysis]
zotero_collection: 2-RL
image_source: online
created: 2026-07-14
---

# GSPO vs SAPO vs SAO — 三种 GRPO 改进算法的深度对比

## 概述

这三种算法都是对 [[DeepSeekMath|GRPO]] 的改进，都来自阿里/清华等中国团队，但解决了 **不同的问题**。

| 维度 | [[GSPO]] | [[SAPO]] | [[SAO]] |
|------|:--------:|:---------:|:--------:|
| 团队 | Qwen Team (阿里) | Qwen Team (阿里) | 清华/Z.AI |
| 时间 | 2025.07 | 2025.12 | 2026.07 |
| **核心问题** | Token 级 IS **理论错误** | 硬裁剪造成**不稳定** | 异步 RL 中 GRPO **不匹配** |
| **解决方案** | 序列级 IS ratio | 软门控函数 | 单 rollout + DIS |
| **算法变化** | 💡 最小改动 | ✅ 中等改动 | 🔄 架构级改动 |

---

## 1. 问题定位不同

### GSPO: 理论层面
> GRPO 的 token 级重要性采样**从根本上就是错的**

- 重要性采样要求每个分布有**多个样本**
- Token 级每个位置只有 1 个样本 → 不做分布校正
- 长序列噪声累积 → 不可逆崩溃

**核心洞察**: 奖励是序列级的，优化也应该是序列级的

### SAPO: 优化层面
> 硬裁剪的"全有或全无"过于粗糙

- Token 级 importance ratio 方差高，MoE 下更严重
- 硬裁剪要么保留全部梯度要么归零 → 浪费信号
- 负优势时梯度扩散到大量无关 token

**核心洞察**: 需要平滑的信任域，而非硬边界

### SAO: 系统层面
> GRPO 的组采样设计与异步 RL 不兼容

- 同步 GRPO 每组 8 个 rollout，等最慢的 → 离策略偏差
- 异步场景无法存 `π_old` 检查点 → 需要新方案
- Agentic RL 多轮交互中组采样不现实

**核心洞察**: 异步 RL 需要重新设计训练流程

---

## 2. 解决方案对比

### GSPO: 改 Importance Ratio

$$\text{GRPO: } \underbrace{\frac{\pi_\theta(y_t)}{\pi_{\theta_{\text{old}}}(y_t)}}_{\text{每个 token 各自比}} \quad\Rightarrow\quad \text{GSPO: } \underbrace{\left(\frac{\pi_\theta(y)}{\pi_{\theta_{\text{old}}}(y)}\right)^{1/|y|}}_{\text{整个序列一个比}}$$

- 同一序列所有 token **等权重**
- 裁剪比例 0.13% → **15%**（但更高效！）
- MoE 无需 Routing Replay
- 可直接用推理引擎的似然

### SAPO: 改裁剪函数

$$\text{GSPO: } \min(s\hat{A}, \text{clip}(s)\hat{A}) \quad\Rightarrow\quad \text{SAPO: } \underbrace{\sigma(\tau(r-1))\cdot\frac{4}{\tau}}_{\text{平滑门控}}\cdot\hat{A}$$

- r=1 时梯度完全保留，偏离越远衰减越平滑
- 非对称温度：$\tau_{neg}=1.05 > \tau_{pos}=1.0$
- 负优势 token 衰减更快（负梯度扩散到大量词汇）
- MoE 也无需特殊处理

### SAO: 改训练流程（幅度最大）

$$\text{GRPO: } \underbrace{\text{采 8 个 → 等最慢 → 一起训练}}_{\text{同步组采样}} \quad\Rightarrow\quad \text{SAO: } \underbrace{\text{采 1 个 → 立即训练}}_{\text{异步单 rollout}}$$

**双面 Token 级裁剪 (DIS)**:
$$f(x; \epsilon_l, \epsilon_h) = \begin{cases}x, & 1-\epsilon_l < x < 1+\epsilon_h \\ 0, & \text{otherwise}\end{cases}$$

**价值模型 3 技巧**:
1. **更快更新**: 价值网络每步 2 次（策略 1 次）
2. **冻结 Attention**: 只训练 MoE 层
3. **Skip-Observation GAE**: 多轮 Agent 任务跳过环境观测

---

## 3. 实验对比

| Benchmark | GRPO | GSPO | SAPO | SAO |
|-----------|:----:|:----:|:----:|:----:|
| AIME2025 | 84.2 | ~+2-3 | ~+2-3 | **97.3** |
| BeyondAIME | 54.8 | — | — | **74.8** |
| HMMT | 76.0 | — | — | **88.3** |
| IMOAnswer | 55.8 | — | — | **74.0** |
| SWE-bench | 27.0 | — | — | **29.8** |
| 训练稳定性 | ❌ 160步崩溃 | ✅ 稳定 | ✅ 稳定 | ✅ **1000步稳定** |
| MoE 特殊处理 | 需 R2 | ✅ 不需要 | ✅ 不需要 | ✅ 不需要 |

> **注**: GSPO 和 SAPO 论文实验在 Qwen3-30B-A3B，SAO 在相同模型。直接数字对比不完全公平，因评估集和时间点不同。

---

## 4. 适用场景指南

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| **同步训练，短响应** (≤4K tokens) | [[GSPO]] | 改动最小，解决核心 IS 问题 |
| **同步训练，MoE 模型** | [[SAPO]] | 软门控 + 非对称温度更适合 MoE |
| **异步训练 / Agentic RL** | [[SAO]] | 唯一为异步设计的方案 |
| **多轮交互 / Agent** | [[SAO]] | Skip-Observation GAE 原生支持 |
| **资源有限，无法存 π_old** | [[SAO]]/[[GSPO]] | 都无需检查点管理 |
| **与现有代码库集成** | [[GSPO]] | 只需改几行 IS ratio 计算 |

---

## 5. 与 StabilizingRL 论文的关系

[[StabilizingRL]]（同 Qwen 团队）为这些改进提供了形式化基础：

- 它证明了 **Token 级目标** 是序列级目标的 **一阶近似**
- 将 IS 权重分解为 **训练-推理不一致性 × 策略过时性**
- 提出 MiniRL 作为最小化基线
- 提出 R2/R3 路由重放

因此正确的理解是：

```
StabilizingRL (形式化基础)
  ├── 理论依据 → GSPO (改 IS ratio)
  ├── 理论依据 → SAPO (改裁剪函数)
  └── 部分借鉴 → SAO (重训练流程)
```

---

## 6. 总结

| | GSPO | SAPO | SAO |
|---|:----:|:----:|:----:|
| **改动量** | 最小 | 中等 | 最大 |
| **创新高度** | ★★★★ | ★★★★ | ★★★★★ |
| **通用性** | 高 | 高 | 中（专为异步） |
| **MoE 友好** | ✅ | ✅ | N/A |
| **异步兼容** | ❌ | ❌ | ✅ |
| **代码改动** | ~5行 | ~20行 | 重写训练循环 |

**一句话选型**:
- 如果你是**改动越少越好** → **GSPO**
- 如果模型是 **MoE 且训练不稳定** → **SAPO**
- 如果要训练 **Agentic / 多轮交互** → **SAO**

---

*整理时间: 2026-07-14*
