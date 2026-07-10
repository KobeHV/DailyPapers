---
title: "Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning"
method_name: "SAO"
authors: [Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong]
year: 2026
venue: arXiv
tags: [rl, grpo, async-training, llm, post-training, agent, off-policy, importance-sampling, value-model, gae]
zotero_collection: 
image_source: online
arxiv_html: https://arxiv.org/html/2607.07508
created: 2026-07-09
---

# 论文笔记：Single-Rollout Asynchronous Optimization for Agentic RL

## 元信息

- **完整标题**: Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning
- **作者**: Zhenyu Hou\*, Yujiang Li\*, Jie Tang, Yuxiao Dong (\*equal contribution)
- **机构**: Tsinghua University (work done while interning at Z.AI / 智谱AI)
- **发布日期**: 2026-07-08 (arXiv)
- **状态**: Under review
- **arXiv ID**: 2607.07508
- **项目页面**: [papers.cool/arxiv/2607.07508](https://papers.cool/arxiv/2607.07508)
- **生产部署**: 已部署于 GLM-5.2 (750B-A40B) 模型的 agentic RL 训练管线

## 一句话总结

> 用单次采样替代 GRPO 的组采样实现稳定异步 RL 训练。

## 核心贡献

1. **Single-Rollout Sampling**: 提出用每条 prompt 仅一次 rollout（而非 GRPO 的 group-wise 多次采样）来适配异步 agentic RL 训练，消除"straggler"瓶颈，减少 off-policy 效应。
2. **DIS (Direct Double-Sided Importance Sampling) Clipping**: 设计严格的双边 token 级重要性采样裁剪机制，在区间 $[1-\epsilon_\ell, 1+\epsilon_h]$ 外直接将 token 从梯度计算中 mask 掉，应对异步训练中的 policy lag 问题。
3. **Value Model 训练设计**: 重新引入参数化价值模型作为 critic，配合 (a) 高频价值更新 (K=2), (b) Frozen-Attention Regularization, (c) Skip-Observation Token-Level GAE，使单 rollout 训练实用化。
4. **生产级验证**: 在 GLM-5.2 (750B 总参数, 40B 激活参数) 的 agentic RL post-training 管线中成功部署，证明了方法在大规模生产环境中的可行性。

## 问题背景

### 要解决的问题

LLM 的 post-training 阶段越来越多地使用强化学习 (RL)。然而，针对长周期 agentic 任务（如代码修复、多轮工具使用）的 RL 训练面临效率瓶颈：

- **同步 RL (Synchronous RL)** 要求等待一个 batch 中所有 rollout 完成后再更新模型，而 agentic 任务的 rollout 长度差异巨大（可能从数秒到数分钟），导致 GPU 空闲等待"straggler"轨迹。
- **异步 RL (Asynchronous RL)** 理论上可以在每个 rollout 完成时立即更新模型，提升吞吐，但引入了 **policy lag**（生成 rollout 的模型版本与当前训练模型版本不一致）和 **off-policy effects**，导致训练不稳定甚至崩溃。

### 现有方法的局限

**GRPO (Group Relative Policy Optimization)** 是目前最主流的 LLM RL 算法（被 DeepSeek-R1、Qwen3、GLM-5 等广泛使用），但其设计与异步训练存在根本矛盾：

1. **Group-wise Sampling 与异步训练不兼容**: GRPO 需要为每条 prompt 采样多条 response（组内相对优势估计依赖组内均值/方差）。在异步设置中，等待完整 group 会重新引入 straggler 延迟，与异步训练的初衷矛盾。
2. **Agentic 任务通常只提供单轨迹反馈**: 代码修复任务中，编译器只返回一次错误信息；test suite 只返回一次 pass/fail。多次 rollout 往往不可行或不必要。
3. **无价值模型**: GRPO 用 running-mean baseline 替代价值函数，但在异步 off-policy 场景下，running-mean 无法准确跟踪快速变化的策略，导致优势估计偏差严重。

### 本文的动机

设计一种**原生适配异步训练**的 RL 算法，在保持训练稳定性的同时，充分利用异步训练的吞吐优势，并适应 agentic 任务的单轨迹反馈特性。

## 方法详解

### 模型架构总览

SAO 的整体架构包含以下核心组件：

1. **Rollout Engine** (基于 SGLang/vLLM): 异步生成 rollout 轨迹，每个 rollout 完成后立即送入 training buffer。
2. **Actor (Policy Network $\pi_\theta$)**: 从 buffer 中消费单条 rollout 进行梯度更新。
3. **Critic (Value Network $V_\phi$)**: 以 K 倍于 actor 的频率更新，提供稳定的 advantage 基线。
4. **DIS Clipping Module**: 对每条 rollout 的每个 token 计算 importance ratio 并进行双边裁剪/屏蔽。
5. **Skip-Observation GAE**: 针对 agentic 多轮轨迹的专用 advantage 估计器。

```
┌─────────────────────────────────────────────┐
│              Rollout Engine                   │
│  (SGLang/vLLM, async inference)              │
│                                               │
│  Prompt -→ π_rollout -→ trajectory            │
│  (version may lag behind current π_θ)         │
└──────────────────┬──────────────────────────┘
                   │ single rollout per prompt
                   ▼
┌─────────────────────────────────────────────┐
│             Training Buffer                   │
│  stores (s, a, r, log_prob_rollout, ...)     │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
┌──────────────┐         ┌──────────────┐
│  Actor π_θ   │         │  Critic V_ϕ   │
│  (updated    │         │  (updated K×  │
│   every step)│         │   per step)   │
│              │         │               │
│  DIS Clipping│         │  Frozen-Atten │
│  Token-Level │         │  Regularizat. │
└──────────────┘         └──────────────┘
```

### 核心模块

#### 1. Single-Rollout Sampling

**与 GRPO 的关键区别**:

| 方面 | GRPO | SAO |
|------|------|-----|
| 每条 prompt 的采样数 | G 条 response (G≥2, 通常 G=4~16) | 1 条 response |
| 优势估计 | 组内相对优势 $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$ | 价值模型 $V_\phi$ 提供基线 |
| 异步兼容性 | 需等待整组完成 | 到达即处理 |
| Off-policy 程度 | 组内等待加剧 policy lag | 单次采样最小化 lag |

**单 rollout 的优势**:
- **最小化 policy lag**: rollout 生成到被训练消费的时间窗口最短
- **天然适配 agentic 任务**: 许多环境本身只提供单轨迹反馈
- **更好的泛化**: 减少组内 overfitting（GRPO 可能利用组内分布特征作弊）

**单 rollout 的挑战与应对**:
- 无组内归一化 → 优势方差更大 → 需要重新引入价值模型 $V_\phi$
- Off-policy 效应仍需处理 → DIS clipping

#### 2. DIS: Direct Double-Sided Importance Sampling Clipping

DIS 是 SAO 的核心稳定化机制，与标准 PPO clipping 有本质区别。

**标准 PPO Clipping** (one-sided):

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

PPO 的 clipping 是**非对称**的：当 $A>0$ 时只截断 $r_t > 1+\epsilon$（防止过度增加概率），当 $A<0$ 时只截断 $r_t < 1-\epsilon$（防止过度减少概率）。

**SAO 的 DIS Clipping** (double-sided token-level masking):

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{rollout}}(a_t \mid s_t)}$$

$$\text{mask}_t = \begin{cases} 1 & \text{if } 1-\epsilon_\ell \le r_t(\theta) \le 1+\epsilon_h \\ 0 & \text{otherwise} \end{cases}$$

$$L^{\text{DIS}}(\theta) = -\frac{1}{\sum_t \text{mask}_t} \sum_t \text{mask}_t \cdot r_t(\theta) \cdot \hat{A}_t$$

**关键创新点**:

1. **行为策略代理**: 直接使用 rollout 时的 log-probability $\pi_{\text{rollout}}(a_t|s_t)$ 作为分母，无需存储历史模型 checkpoint。这比标准 PPO 的 $\pi_{\theta_{\text{old}}}$ 更直接。

2. **双边严格屏蔽**: 不在 $[1-\epsilon_\ell, 1+\epsilon_h]$ 内的 token **完全排除**出梯度计算（而非仅 clip）。这对两个方向一视同仁，防止 asynchronous 场景下 extreme policy divergence。

3. **Token 级粒度**: 每个 token 独立判断，而非 sequence 级。这允许局部的 policy alignment 不被全局变化污染。

**DIS 的超参数**:
- $\epsilon_\ell$: 下界 clip ratio (典型值约 0.2)
- $\epsilon_h$: 上界 clip ratio (典型值约 0.2)
- 论文中验证了双边界限的重要性

#### 3. Value Model Training Design

SAO 为了支持单 rollout 的优势估计，重新引入了参数化价值模型 $V_\phi$，并设计了三个关键改进：

**a) Faster Value Updates (高频价值更新)**

对于每 1 次 policy $\pi_\theta$ 的梯度更新，价值模型 $V_\phi$ 进行 $K$ 次更新（$K=2$ 为典型设置）。

**动机**: 单 rollout 设置下，advantage 估计 $\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 的质量高度依赖于 $V_\phi$ 的准确性。policy 快速变化时，$V_\phi$ 需要更快地跟上。

**效果**: Ablation 显示 $K>1$ 显著提升 explained variance，使 critic 能为 actor 提供更准确的梯度信号。AIME2025 上 value-based critic (97.3%) 远超 running-mean baseline (79.8%)。

**b) Frozen-Attention Regularization (冻结注意力层)**

价值模型从预训练 LLM 初始化，但在训练中**冻结所有 attention 层**，只训练 MLP/MoE projection layers。

**动机**: 
- 全参数训练价值模型容易出现 **梯度爆炸**，尤其在复杂推理任务上
- Attention 层携带的语义知识可用于价值预测，无需重新学习
- MoE projection 层的适配足以调整价值预测

**效果**: Ablation 显示 frozen attention 使 critic 的 gradient norm 保持低且稳定，防止训练崩溃。

**c) Skip-Observation Token-Level GAE**

这是 SAO 针对 **agentic 多轮轨迹** 的专用设计。

**问题**: Agentic 轨迹中交替出现 model-generated action tokens 和 environment-generated observation tokens:

```
[prompt] → [model act₁] → [env obs₁] → [model act₂] → [env obs₂] → ...
```

标准 GAE 在 token 级别逐 token 计算 TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。但 environment observation tokens 不是模型生成的——模型不对其负责。

**解决方案**: Skip-Observation GAE 修改 Bellman target，将 action sequence 结束时的 value 直接连接到**下一个** action sequence 开始时的 value，**跳过中间的 observation tokens**：

$$V^*_{\text{target}}(s_{\text{action\_end}}) = r_{\text{step}} + \gamma \cdot \mathbb{E}_{a \sim \text{action\_start}}[V(s_{\text{next\_action\_start}})]$$

这确保了 credit assignment 只发生在模型**实际控制的转换**上。

## 关键公式

### 1. GRPO 目标函数 (Baseline)

$$J_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta) A_i,\; \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) A_i\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中：
- $G$: 每条 prompt 的 response 数量 (group size)
- $o_i$: 第 $i$ 条 response
- $|o_i|$: response 的 token 长度
- $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$: token 级 importance ratio
- $A_i = \frac{r_i - \text{mean}(r_1,...,r_G)}{\text{std}(r_1,...,r_G)}$: 组内标准化优势

**GRPO 的局限**: 组标准化优势 $A_i$ 依赖于 group size $G \ge 2$，在单个 rollout 场景下退化为零。

### 2. SAO 单 rollout 目标函数

$$J_{\text{SAO}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{rollout}}}\left[\frac{1}{|\tau|} \sum_{t=1}^{|\tau|} \text{mask}_t \cdot r_t(\theta) \cdot \hat{A}_t\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中：
- $\tau$: 单条 rollout 轨迹
- $\hat{A}_t$: Skip-Observation GAE 计算出的 token 级 advantage
- $\text{mask}_t \in \{0, 1\}$: DIS 的双边屏蔽指示器

### 3. DIS Clipping Formula (核心公式)

**Probability Ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{rollout}}(a_t \mid s_t)}$$

其中 $\pi_{\text{rollout}}$ 是**实际生成 rollout 时的策略**（可能与当前 $\pi_\theta$ 不同，因为 policy lag），直接使用 rollout engine 记录的 log-probability。

**Double-Sided Token Mask**:

$$\text{mask}_t = \mathbf{1}\left[1 - \epsilon_\ell \le r_t(\theta) \le 1 + \epsilon_h\right]$$

其中 $\mathbf{1}[\cdot]$ 是指示函数。

**Effective Policy Loss**:

$$L_{\text{policy}}(\theta) = -\frac{1}{\sum_t \text{mask}_t} \sum_t \text{mask}_t \cdot r_t(\theta) \cdot \hat{A}_t$$

被 mask 的 token 对梯度贡献为零，相当于从 mini-batch 中剔除。

### 4. Value Loss

$$L_{\text{value}}(\phi) = \frac{1}{|\tau|} \sum_{t=1}^{|\tau|} \left(V_\phi(s_t) - R_t\right)^2$$

其中 $R_t$ 是 Skip-Observation GAE 计算的 Monte Carlo return / TD($\lambda$) target：

$$R_t = \sum_{k=0}^{|\tau|-t} (\gamma\lambda)^k \cdot \delta_{t+k}^{\text{skip}}$$

$\delta_{t}^{\text{skip}}$ 是跳过 observation tokens 后的 TD error。

### 5. Skip-Observation GAE

**标准 TD error**:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**Skip-Observation TD error**:

$$\delta_t^{\text{skip}} = \begin{cases} r_t^{\text{step}} + \gamma V(s_{t'}) - V(s_t) & \text{if } t \text{ is action token ending a step} \\ 0 & \text{if } t \text{ is observation token} \end{cases}$$

其中 $t'$ 是**跳过 observation block 后的下一个 action token 位置**，$\gamma=1$ (LLM 中通常不使用折扣)，$\lambda=1$ (等价于 Monte Carlo returns)。

### 6. 完整训练目标

$$\mathcal{L}_{\text{total}}(\theta, \phi) = L_{\text{policy}}(\theta) + \alpha \cdot L_{\text{value}}(\phi) + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

## 关键图表

### Figure 1: Main Results Comparison

Benchmark comparison of SAO vs GRPO vs SFT baseline across 5 benchmarks (AIME2025, BeyondAIME, HMMT Nov 2025, IMOAnswerBench, SWE-Bench Verified), using Qwen3-30B-A3B as base model.

![](https://arxiv.org/html/2607.07508/figures/fig1.png)

### Figure 2: Training Stability Curves

Comparison of training dynamics between SAO (stable for 1000+ steps) vs GRPO variants (often collapse after few hundred steps in async setting).

![](https://arxiv.org/html/2607.07508/figures/fig2.png)

### Figure 3: Ablation Studies

Ablation results on key design choices: (a) double-sided vs single-sided clipping, (b) K value for critic update frequency, (c) frozen-attention vs full-parameter critic, (d) Skip-Observation GAE vs standard GAE.

![](https://arxiv.org/html/2607.07508/figures/fig3.png)

### Figure 4: Online Adaptation Experiment

Performance on simulated online learning with dynamically shifting reward preferences (e.g., writing style preference reversal), showing SAO's rapid policy realignment compared to baselines.

![](https://arxiv.org/html/2607.07508/figures/fig4.png)

### Table 1: Main Results

| Benchmark | SFT Baseline | GRPO | **SAO** |
|-----------|:-----------:|:----:|:-------:|
| AIME2025 | 80.4% | 84.2% | **97.3%** |
| BeyondAIME | 53.3% | 54.8% | **74.8%** |
| HMMT Nov 2025 | 75.2% | 76.0% | **88.3%** |
| IMOAnswerBench | 53.3% | 55.8% | **74.0%** |
| SWE-Bench Verified | 23.0% | 27.0% | **29.8%** |

所有推理 benchmark (AIME2025, BeyondAIME, HMMT, IMOAnswerBench) 在 reasoning-with-Python-tool 设置下评估。基础模型: Qwen3-30B-A3B (MoE)。

### Table 2: Ablation Results

| 消融变体 | AIME2025 | 说明 |
|---------|:--------:|------|
| SAO (full) | **97.3%** | 完整方法 |
| w/o DIS (standard PPO clip) | 显著下降 | 双边裁剪对 off-policy 稳定性关键 |
| w/o Faster Value Updates (K=1) | 显著下降 | 高频 critic 更新提升 explained variance |
| w/o Frozen Attention | 训练崩溃 | 全参数 value 训练导致梯度爆炸 |
| w/o Skip-Observation GAE | 下降 | 在 agentic 任务上 noise 增多 |
| Running-Mean Baseline (no $V_\phi$) | 79.8% | 价值模型对单 rollout 关键 |

## 实验

### 数据集

| Dataset | Domain | Type | Description |
|---------|--------|------|-------------|
| **SWE-Bench Verified** | 代码修复 | Agentic Coding | 500 个真实 GitHub issue 修复任务，含 test suite 验证 |
| **AIME2025** | 数学推理 | Reasoning + Python Tool | 2025 AIME 竞赛题目，使用 Python 工具辅助求解 |
| **BeyondAIME** | 数学推理 | Reasoning + Python Tool | 难度超越 AIME 的数学推理题 |
| **HMMT Nov 2025** | 数学推理 | Reasoning + Python Tool | Harvard-MIT Math Tournament 2025 年 11 月 |
| **IMOAnswerBench** | 数学推理 | Reasoning + Python Tool | IMO 级别数学题的答案评测 |

### 实现细节

| 项目 | 详情 |
|------|------|
| **基础模型** | Qwen3-30B-A3B (MoE, 30B total / 3B active) |
| **RL 框架** | Slime (Megatron-LM + SGLang) |
| **训练步数** | ~1000 步（稳定训练） |
| **硬件** | 8×H100/H800 单节点（单 rollout 设置）；多节点用于生产部署 |
| **推理精度** | FP8 推理 / BF16 训练 |
| **并行策略** | TP=4, EP=8, PP=1 |
| **优化器** | Adam with CPU offload |
| **学习率** | (标准 LLM RL 设置) |
| **K (critic update freq)** | 2 |
| **$\epsilon_\ell, \epsilon_h$ (DIS bounds)** | 典型值 0.2 |
| **$\gamma$ (discount)** | 1.0 (LLM 中通常无折扣) |
| **$\lambda$ (GAE)** | 1.0 (Monte Carlo returns) |
| **KL 系数 $\beta$** | 0.001 (典型值) |
| **生产部署** | GLM-5.2 750B-A40B, post-training 全流程 ~2 天 |

### Main Results

SAO 在所有 5 个 benchmark 上全面超越 SFT baseline 和 GRPO：

1. **AIME2025**: 97.3% (+16.9 over SFT, +13.1 over GRPO) —— 近乎饱和
2. **BeyondAIME**: 74.8% (+21.5 over SFT, +20.0 over GRPO) —— 最大提升
3. **HMMT Nov 2025**: 88.3% (+13.1 over SFT, +12.3 over GRPO)
4. **IMOAnswerBench**: 74.0% (+20.7 over SFT, +18.2 over GRPO)
5. **SWE-Bench Verified**: 29.8% (+6.8 over SFT, +2.8 over GRPO)

**关键观察**:
- SWE-Bench 的提升较小但稳定（agentic coding 任务难度极高，SOTA 也仅 ~75%）
- 推理任务上 SAO 的优势尤为显著，体现了价值模型 + DIS 在复杂推理 chain 上的优势
- Running-mean baseline 仅达 79.8% (AIME2025)，与 SAO 的 97.3% 差距巨大，证明价值模型对单 rollout 训练不可或缺

### Ablation Studies

1. **DIS vs Standard PPO Clip**: 标准 PPO clipping 在异步场景下不稳定，双边严格 masking 是关键。
2. **Faster Value Updates (K=2 vs K=1)**: K=2 显著提升 explained variance，K>2 收益递减。
3. **Frozen-Attention Regularization**: 全参数训练 critic 导致梯度爆炸和训练崩溃；冻结 attention 是训练稳定性的必要条件。
4. **Skip-Observation GAE**: 在 SWE-Bench 等 agentic 任务上有明显增益，在纯文本推理任务上增益较小（因为没有 environment observation tokens）。
5. **Value Model vs Running-Mean**: 价值模型在单 rollout 设置下大幅度优于 running-mean baseline。

### Training Dynamics

- **SAO**: 稳定训练 1000+ 步，reward 持续上升，无明显坍塌迹象。
- **GRPO (async)**: 通常在数百步后出现 reward 骤降 / 训练崩溃。
- **在线适应实验**: 在 reward preference 动态变化（如写作风格偏好反转）的模拟环境中，SAO 展现出快速策略重对齐能力，恢复速度显著快于 GRPO。

### 在线适应实验 (Online Adaptation)

论文包含一个**模拟在线学习**实验：
- 环境动态改变 reward preference（如从偏好"简洁风格"切换到"详细风格"）
- SAO 的单 rollout 策略能快速感知 reward 变化并调整策略
- GRPO 的 group-wise 采样在环境变化时反应更慢（需等待完整 group）
- 实验验证了 SAO 在**非平稳环境**中的优越性

## 批判性思考

### 优点

1. **原生异步设计**: 与 GRPO 的同步 batch 设计形成鲜明对比，SAO 从第一性原理出发解决异步 RL 的不稳定问题。
2. **生产验证**: 不是纯学术探索，已在 GLM-5.2 750B 模型的实际训练中验证了可行性。
3. **系统性工程**: 不仅提出了算法创新（DIS, Skip-Observation GAE），还解决了配套的工程挑战（value model 训练稳定性，异步 pipeline 设计）。
4. **结果扎实**: 5 个 benchmark 全面超越，ablation 完整，在线适应实验额外验证。
5. **与 Slime 框架深度整合**: 作为一个完整的异步 RL 训练解决方案，而非孤立的算法。

### 局限性

1. **价值模型的计算开销**: 相比 GRPO（无需价值模型），SAO 需要额外的 critic 训练，增加了显存和计算开销。这在论文中提到但未详细量化。
2. **SWE-Bench 提升有限**: 29.8% 虽然超过 GRPO，但距离 SOTA (~75%) 仍有很大差距。SAO 主要解决的是训练稳定性而非 agent 能力上限。
3. **超参数敏感度**: DIS 的双边 clip bounds ($\epsilon_\ell, \epsilon_h$), K 值, frozen-attention 等组合较多，不同任务可能需要不同配置。
4. **缺乏与其他异步方法的全面对比**: 仅与 GRPO 及其变体比较，未与同期异步 RL 方法（VCPO, M2PO, GEPO 等）做系统对比。
5. **单 rollout 方差的潜在问题**: 虽然价值模型弥补了方差，但在高随机性环境（如某些创造性写作任务）中，单 rollout 采样可能不够。

### 潜在改进方向

1. **自适应 DIS bounds**: $\epsilon_\ell, \epsilon_h$ 可根据 policy lag 的程度动态调整。
2. **与 GAE($\lambda$) 的更深度整合**: 论文已经用了 Skip-Observation GAE，可以进一步探索 $\lambda$ 的自适应方案。
3. **multi-agent 扩展**: 将 SAO 应用于 multi-agent 异步训练场景。
4. **更高效的价值模型**: 探索更轻量的 critic 架构（如 LoRA-based critic）以减少开销。
5. **与其他异步方法的组合**: DIS clipping 可以与 VCPO 的方差控制、GEPO 的 group-expectation 等方法结合。

### 可复现性评估

- **开源框架**: 基于开源的 Slime 框架，训练脚本公开 (github.com/THUDM/slime)
- **模型可用**: Qwen3-30B-A3B 开源可获取
- **Benchmark 公开**: 所有 5 个 benchmark 均为公开数据集
- **超参数**: 论文提供了关键超参数，但完整的复现配置需要从 Slime 仓库中获取
- **计算门槛**: 单节点 8×H100/H800 可运行，门槛相对可控
- **总体评估**: 可复现性较高

## 关联笔记

### 基于

- [[GRPO]]: SAO 的设计出发点是解决 GRPO 在异步场景下的不兼容问题；SAO 保留了 GRPO 的 token-level PPO-style objective，但将组标准化优势替换为价值模型基线
- [[PPO]]: DIS clipping 继承自 PPO 的 clipping 思想，但做了双边严格化的扩展

### 对比

- [[VCPO]] (ICML 2026): 方差控制的 off-policy RL，使用 ESS 引导的步长缩放
- [[M2PO]]: 约束 importance weight 的二阶矩而非逐 token clip
- [[GEPO]]: 用 group-expectation weight 替代 per-sample weight
- [[DAPO]]: GRPO + 非对称 clipping ($\epsilon_{\text{low}} \neq \epsilon_{\text{high}}$) + 动态采样

### 方法相关

- [[CompactionRL]]: 同期工作 (arXiv:2607.05378)，同样用于 GLM-5.2 训练，关注 context compaction for long-horizon agents
- [[T1]]: 同一作者团队的早期工作，使用 oversampling + token-entropy regularization 做推理 RL
- [[Slime]]: SAO 所基于的开源 RL 训练框架

### 硬件/数据相关

- [[GLM-5]]: SAO 部署的模型系列
- [[SWE-Bench Verified]]: agentic coding benchmark
- [[AIME]]: 数学推理 benchmark

## 速查卡片

| 项目 | 内容 |
|------|------|
| **全称** | Single-Rollout Asynchronous Optimization |
| **核心思想** | 单 rollout + 双边 importance sampling clipping + 价值模型 |
| **与 GRPO 的本质区别** | 组采样 → 单采样；组标准化优势 → 价值模型基线 |
| **DIS** | $r_t(\theta) \in [1-\epsilon_\ell, 1+\epsilon_h]$ 外的 token 直接 mask |
| **价值模型设计** | 高频更新 (K=2) + 冻结 attention + Skip-Observation GAE |
| **AIME2025** | 97.3% (SFT: 80.4%, GRPO: 84.2%) |
| **训练稳定性** | 1000+ 步无崩溃 |
| **生产部署** | GLM-5.2 (750B-A40B) |
| **框架** | Slime (Megatron + SGLang) |
| **基础模型** | Qwen3-30B-A3B (实验) / GLM-5.2 (生产) |
