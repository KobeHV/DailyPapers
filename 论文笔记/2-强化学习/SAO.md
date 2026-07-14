---
title: "Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning"
method_name: "SAO"
authors: [Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong]
year: 2026
venue: arXiv
tags: [reinforcement-learning, asynchronous-rl, agentic, single-rollout, glm, dis]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2607.07508v1
created: 2026-07-14
updated: 2026-07-14
aliases: [SAO, Single-Rollout Asynchronous Optimization, DIS]
---

# 论文笔记：Single-Rollout Asynchronous Optimization (SAO)

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | Tsinghua University / Z.AI |
| **作者** | Zhenyu Hou*, Yujiang Li*, Jie Tang, Yuxiao Dong（* 同等贡献，在 Z.AI 实习期间完成） |
| **发表** | arXiv:2607.07508, 2026年7月 |
| **对比基线** | GRPO, VAPO, GRPO+DIS, SFT |
| **部署模型** | **GLM-5.2 (750B-A40B)** |
| **链接** | [arXiv](https://arxiv.org/abs/2607.07508) |

---

## 一句话总结

> **单 Rollout 异步 RL** — 解决 GRPO 组采样与异步训练的根本不兼容，配合双面 DIS 裁剪和 Skip-Observation GAE，1000 步稳定训练，已用于 GLM-5.2 (750B)。

---

## 问题背景

### 同步 RL 的问题

之前的 LLM RL 都是**同步、批交错**训练：

```
GRPO 同步流程:
  1. 采样 G=8 个响应 / prompt
  2. 等最慢的响应完成
  3. 一起训练更新

问题:
  - GPU 空转等待慢响应
  - 长 Agent 任务差异巨大（毫秒级 → 分钟级）
  - 不适合实时交互
```

### 异步 RL 的挑战

异步 RL 更高效（完成即训练），但面临两个挑战：

**挑战 1: 离策略 (Off-Policy)**
- 异步流水线中，旧模型 rollouts 和新模型参杂
- GRPO 的组等待加剧离策略程度

**挑战 2: GRPO 组采样不兼容**
- GRPO 需要同时执行 G 个响应 → 异步流水线无法做组同步
- 即使勉强实现，组内奖励归一化在异步环境下有偏

---

## 方法详解

### SAO 的核心思路：单 Rollout

> 每 prompt 只生成 **1 个 rollout**，完成后立即送入训练

```
GRPO 的组采样:      [========采样 G 个========] [===等最慢===] [训练]
SAO 的单 rollout:   [=采样 1 个=] [训练] [=采样 1 个=] [训练] ...
```

### SAO 优化目标

$$\mathcal{L}(\theta) = \hat{\mathbb{E}}_t\left[f(r_t(\theta), \epsilon_l, \epsilon_h)\hat{A}_t \log \pi_\theta(a_t|s_t)\right]$$

其中 **DIS (Direct Importance Sampling)** 的 importance ratio:

$$r_t(\theta) = \exp\left(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{rollout}}(a_t|s_t)\right)$$

**关键**: 抛弃 $\pi_{\theta_{\text{old}}}$！直接用 $\pi_{\text{rollout}}$ 的 log-prob：

> 异步环境中无法追踪旧策略 $\pi_{\theta_{\text{old}}}$（模型可能已经更新多次）。
> SAO 直接用 rollout 时的 log-prob 计算 ratio，无需存储检查点。

### 双面 Token 级裁剪 (DIS)

**PPO 的裁剪**（单面，取决于优势符号）：

$$\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$$

**SAO 的双面裁剪**（梯度掩码，不是 clip 是 mask！）：

$$f(x; \epsilon_l, \epsilon_h) = \begin{cases} x, & 1-\epsilon_l < x < 1+\epsilon_h \\ 0, & \text{otherwise} \end{cases}$$

| 方面 | PPO Clip | SAO DIS |
|:----|:--------:|:-------:|
| 裁剪方式 | min + clip（保留边界值） | **完全 masked**（梯度归零） |
| 方向 | 取决于优势符号 | **对称双面** |
| 严格程度 | 温和 | **严格** |
| 超参数 | $\epsilon=0.2$ | $\epsilon_l=0.3, \epsilon_h=5.0$（数学） |

### 价值模型训练技巧

**技巧 1: 更快更新 (K=2)**
- 每 1 次策略更新，价值网络更新 **2 次**
- 帮助价值模型快速追踪变化的策略分布

**技巧 2: 冻结 Attention**
- 价值模型的 **Attention 层冻结**，只训练 MoE 层
- 动机：价值模型梯度范数远大于策略模型，不稳定源于 Attention 层

| 方法 | 梯度范数 | 稳定性 |
|:----|:--------:|:------:|
| Full-parameter | 高 | ❌ |
| **Frozen Attention** | **低** | **✅** |

**技巧 3: Skip-Observation GAE**

多轮 Agent 轨迹的问题：$[a_0, o_0, a_1, o_1, \dots]$

标准 GAE 在 action→observation 边界计算 value diff 时不正确（observation 是环境输出，不是模型生成的）。

$$\hat{A}(a_{i,N}) = \delta + \gamma\lambda\hat{A}(a_{i+1,0})$$
$$\delta = r_t + \gamma V(a_{i+1,0}) - V(a_{i,N})$$

即跳过 observation token，直接从 action i 的最后 token 跳到 action i+1 的第一个 token。

### 完整算法伪代码

```
1: 初始化 π_θ, V_φ
2: for step = 1 to N do:
3:     对每个 prompt，采样 1 个 rollout
4:     保存 rollout 的 log-prob (用于 IS ratio)
5:     计算优势（使用 Skip-Observation GAE）
6:     for 策略更新 (×1):
7:         计算 DIS 裁剪后的 SAO 目标
8:         梯度更新 π_θ
9:     for 价值更新 (×2):
10:        冻结 Attention，只训练 MoE 层
11:        两次梯度更新 V_φ
12: end for
```

---

## 实验结果

### 主实验设置

| 项目 | 配置 |
|:----|------|
| 模型 | Qwen3-30B-A3B-Base |
| 数学裁剪 | $\epsilon_l=0.3, \epsilon_h=5.0$ |
| 代码裁剪 | $\epsilon_l=0.8, \epsilon_h=3.0$ |
| 最大训练步 | **1000** |

### 数学推理基准

| 方法 | AIME2025 | BeyondAIME | HMMT Nov 2025 | IMOAnswerBench |
|:----|:--------:|:----------:|:-------------:|:--------------:|
| Claude-Sonnet-4.5 | 87.0 | 62.0 | 81.7 | 65.8 |
| GPT-5 High | 94.6 | 74.0 | 89.2 | 76.0 |
| Qwen3-30B (SFT, w/ python) | 80.4 | 53.3 | 75.2 | 53.3 |
| GRPO (w/ python) | 84.2 | 54.8 | 76.0 | 55.8 |
| GRPO + DIS | 93.5 | 70.8 | 84.0 | 70.0 |
| **SAO** | **97.3** | **74.8** | **88.3** | **74.0** |

> SAO 在所有基准上超越 GRPO，GRPO 在 ~160 步后崩溃，SAO 稳定训练 **1000 步**。

### 消融分析

| 变体 | AIME2025 | BeyondAIME |
|:----|:--------:|:----------:|
| **SAO (完整)** | **97.3** | **74.8** |
| w/o Faster value update | 95.0 | 69.8 |
| w/o Frozen attention | 90.6 | 74.5 |
| Vanilla VAPO (w/o DIS) | 91.3 | 69.0 |
| Running mean baseline | 79.8 | 55.3 |

### SWE-Bench Verified

| 方法 | 准确率 |
|:----|:------:|
| Qwen3-30B-A3B (SFT) | 23.0% |
| + GRPO (w/ DIS) | 27.0% |
| **+ SAO** | **29.8%** |

### 动作粒度消融

| 动作粒度 | AIME2025 | BeyondAIME |
|:---------|:--------:|:----------:|
| 步骤级（平均） | 85.8 | 60.5 |
| 步骤级（最后 token） | 87.3 | 62.8 |
| **Token 级 (SAO)** | **89.8** | **66.8** |

### 在线学习模拟

设计 3 种写作风格（可爱风、中二风、古典风），奖励标准随机切换：

- SAO 在每次偏好切换后**快速适应**
- 对比 Running Mean 基线：SAO **恢复更快、稳定水平更高**

### 训练动态分析

| 指标 | GRPO | SAO |
|:----|:----:|:----:|
| Explained Variance | 低（价值网络差） | **高** |
| Critic 梯度范数 | 高 | **低**（冻结 attention） |
| Clip Ratio | 接近 0（VAPO 崩溃） | **持续活跃** |

---

## GLM-5.2 部署

SAO 已成功部署于 **GLM-5.2 (750B-A40B)** 的 agentic RL 训练流水线，证明了方法的工程可行性。

---

## GRPO vs SAO 详细对比

| 维度 | GRPO | SAO |
|:----|:----:|:----:|
| **采样方式** | 组采样 (G=8) | **单 rollout** |
| **训练时机** | 组全部完成 | **立即训练** |
| **价值网络** | ❌ 不需要 | ✅ 需要 (含冻结 Attention) |
| **IS ratio** | 基于 $\pi_{\theta_{\text{old}}}$ | **基于 $\pi_{\text{rollout}}$** |
| **裁剪** | 单面 CLIP | **双面 DIS (mask)** |
| **GAE** | 标准 | **Skip-Observation** |
| **适用场景** | 同步训练 | **异步 + Agentic** |
| **最大稳定步数** | ~160 步 | **>1000 步** |

---

## 批判性分析

### 优点
1. **异步原生设计**: 专为异步 RL 设计，不是同步方法的修补
2. **1000 步稳定**: 验证了长时间训练的稳定性
3. **工程落地**: 已用在 750B 模型训练中
4. **通用性强**: 数学推理 + 编程 + 在线学习全覆盖

### 局限性
1. **依赖价值模型**: SAO 需要良好的价值模型（GRPO 不需要）
2. **单 rollout 的高方差**: 单样本方差高于组采样（通过价值模型补偿）
3. **只验证 Qwen3**: 未在更小/更大的模型上验证
4. **异步基础设施要求**: 需要保存 rollout log-prob，增加存储和通信

---

## 关联笔记

### 同领域对比
- [[GSPO]] — 同步训练，序列级 IS ratio
- [[SAPO]] — 同步训练，软门控
- [[StabilizingRL]] — 形式化框架 + R2/R3
- [[DeepSeekMath|GRPO]] — SAO 改进的起点

---

## 速查卡片

> [!summary] SAO
> - **核心**: 单 rollout + 双面 DIS + 异步训练
> - **DIS**: $f(x) = x$ if $1-\epsilon_l < x < 1+\epsilon_h$, else $0$
> - **价值模型**: K=2 更新、冻结 Attention、Skip-Observation GAE
> - **结果**: AIME2025 **97.3%** (GRPO 84.2%)，1000 步稳定
> - **部署**: GLM-5.2 (750B-A40B)
> - **影响**: 首个专为异步 Agentic RL 设计的 LLM 训练算法

---

*笔记创建时间: 2026-07-14 | 深度版*
