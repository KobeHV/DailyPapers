---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
method_name: "DPO"
authors: [Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn]
year: 2024
venue: NeurIPS 2024
tags: [preference-optimization, rlhf, alignment, reward-model-free, classification, theory]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2305.18290v1
created: 2026-07-14
updated: 2026-07-14
aliases: [Direct Preference Optimization, DPO]
---

# 论文笔记：Direct Preference Optimization (DPO)

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | Stanford University (IRIS Lab) |
| **作者** | Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn |
| **发表** | NeurIPS 2024, arXiv:2305.18290 |
| **引用量** | > 5,000 |
| **对比基线** | PPO-based RLHF, Preferred-FT, Unlikelihood, Best-of-N |
| **链接** | [arXiv](https://arxiv.org/abs/2305.18290) \| [GitHub](https://github.com/eric-mitchell/direct-preference-optimization) |

---

## 一句话总结

> 通过变量变换将 RLHF 的奖励建模 + RL 优化两阶段**压缩为一步分类问题**，证明语言模型本身就是隐式奖励模型。

---

## 问题背景

### 要解决的问题
如何高效地将语言模型与人类偏好对齐？

### RLHF 的三阶段范式及其问题

标准 RLHF 流程：

```
Stage 1: SFT ─── 在高质量数据上监督微调基础模型
Stage 2: RM  ─── 训练奖励模型拟合人类偏好
Stage 3: RL  ─── 用 PPO 优化策略以最大化奖励（同时约束 KL 散度）
```

**RLHF 的三个痛点**：
1. **复杂度高**: 需要同时维护策略模型、参考模型、奖励模型、价值网络（4 个模型！）
2. **不稳定**: PPO 对超参数敏感，需要精心调整，容易崩溃
3. **计算昂贵**: 训练奖励模型需要额外数据和计算，RL 阶段需要 on-policy 采样

### 本文的核心洞察

> RLHF 的约束奖励最大化问题可以用一个**闭式解**表达—通过变量变换，我们可以直接通过一个简单的分类损失来优化策略。

---

## 方法详解

### 理论推导

#### Step 1: RLHF 的约束优化问题

标准 RLHF 目标（在 KL 约束下最大化奖励）：

$$\max_{\pi_\theta} \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(y|x)}[r_\phi(x,y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) \parallel \pi_{\text{ref}}(y|x)]$$

其中 $\beta$ 控制与参考模型的偏离程度。

#### Step 2: 闭式解

这个 KL 约束优化问题的**闭式最优策略**为：

$$\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$ 是配分函数。

#### Step 3: 反解奖励函数

**关键一步**：将上式变形，用策略表示奖励：

$$r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

#### Step 4: 代入 Bradley-Terry 偏好模型

Bradley-Terry 模型假设人类偏好分布为：

$$p^*(y_1 \succ y_2 | x) = \frac{\exp(r(x,y_1))}{\exp(r(x,y_1)) + \exp(r(x,y_2))} = \sigma(r(x,y_1) - r(x,y_2))$$

将 Step 3 的奖励代入：

$$p^*(y_1 \succ y_2 | x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} - \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}\right)}$$

> **$Z(x)$ 奇迹般地消掉了** — 因为 BT 模型只取决于两个响应的**奖励差值**。

#### Step 5: DPO 损失函数

现在我们可以直接最大化偏好数据的似然：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

其中：
- $x$：提示
- $y_w$：被偏好的（胜利）回答
- $y_l$：不被偏好的（失败）回答
- $\pi_\theta$：正在优化的策略
- $\pi_{\text{ref}}$：参考策略（通常为 SFT 模型）
- $\beta$：控制与参考策略偏离的超参数
- $\sigma$：logistic sigmoid 函数

### 梯度分析

**DPO 的梯度**揭示了其学习机制：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[
\underbrace{\sigma\left(\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}\right)}_{\text{权重：模型越"错"权重越大}}
\left(
\underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\uparrow\ \text{提高获胜回答概率}}
-
\underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\downarrow\ \text{降低失败回答概率}}
\right)
\right]$$

**权重项的关键行为**：
- 当模型**正确排序**（$\pi_\theta(y_w) \gg \pi_\theta(y_l)$）→ 权重接近 0 → **几乎不更新**
- 当模型**排序错误**（$\pi_\theta(y_w) \leq \pi_\theta(y_l)$）→ 权重接近 1 → **大幅更新**
- 隐含奖励 $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$

> 这意味着 DPO 自动加权—它将梯度集中在模型犯错最严重的样本上，类似于 hard negative mining。

### 隐式奖励函数

DPO 定义**隐式奖励函数**：

$$\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

| 方面 | RLHF 显式奖励模型 | DPO 隐式奖励 |
|------|:-----------------:|:-------------:|
| 需要额外模型 | ✅ 是 | ❌ 否 |
| 训练方式 | 独立训练 RM | 策略参数定义 |
| 可解释性 | 标量输出 | 可通过 log-ratio 分析 |

### 理论保证

**定理 1**: 在 Bradley-Terry 偏好模型下，所有与数据一致的奖励函数类都可以用重参数化 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 表示。

> 证明构造了一个投影算子 $f(r; \pi_{\text{ref}}, \beta)(x,y) = r(x,y) - \beta \log \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$

**引理 1**: 同一等价类的不同奖励函数（差一个仅与 x 有关的项）诱导相同的偏好分布。

**引理 2**: 同一等价类的不同奖励函数诱导相同的约束最优策略。

### PPO vs DPO 的等价性分析

论文证明 PPO 的 RLHF 目标可以改写为：

$$\mathcal{J}_{\text{PPO-RLHF}} = \mathbb{E}\left[\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)\right] - \beta^{-1}\mathbb{E}\left[\log Z(x)\right] + \text{const}$$

其中 $Z(x)$ 的估计需要价值函数，这就是 PPO 不稳定的根源。DPO 通过变量变换天然回避了这个问题。

---

## 实验设计与结果

### 实验 1：受控情感生成 (IMDB)

| 方法 | 奖励-KL 前沿 |
|------|:------------:|
| **DPO** | **严格主导**所有基线 |
| PPO (ground-truth reward) | 次优 |
| PPO (learned reward) | 更差 |
| Preferred-FT | 有限 |

> DPO 在任何 KL 散度水平下取得最高奖励。

### 实验 2：摘要 (Reddit TL;DR)

| 方法 | GPT-4 Win Rate (temp=0) |
|------|:-----------------------:|
| **DPO** | **~61%** |
| PPO | ~57% |
| Best-of-N (N=128) | ~58% |
| 人类参考摘要 | 基线 |

**关键发现**: DPO 对**采样温度最鲁棒**：
- DPO：温度 0~1 之间性能稳定
- PPO：高温下性能退回到 GPT-J 基线水平

**人类评估**: DPO (temp=0.25) 被人类评估者偏好 **58%** 超过 PPO (temp=0)

### 实验 3：单轮对话 (Anthropic HH)

| 方法 | Pythia-2.8B | 备注 |
|------|:-----------:|------|
| **DPO** | **唯一有效改进** | 超越参考回答 |
| PPO | 未超越基线 | 任何温度都不行 |
| Best-of-128 | 有效但昂贵 | 需要 128× 计算 |

### 人类评判验证

| 对比 | 一致性 |
|------|:------:|
| 人类 vs 人类 | 65-87% |
| GPT-4 vs 人类 | **67-86%** |

> GPT-4 与人类的一致性接近或超过人类之间的一致性，验证了 GPT-4 作为评估代理的合理性。

---

## 超参数经验指南

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| $\beta$ | 0.1~0.5 | 越小学习越快但可能过拟合 |
| 学习率 | 1e-6~5e-6 | 比 SFT 更小的学习率 |
| 优化器 | AdamW | 标准选择 |
| Batch Size | 32~128 | 取决于任务 |
| Epochs | 1~3 | 过多导致过拟合 |

---

## 批判性分析

### 优点
1. **极度简化**: 将 RLHF 从 3 阶段压缩为 1 阶段，代码量从数千行减少到数十行
2. **训练稳定**: 无需 on-policy 采样、奖励归一化、KL 调参
3. **计算高效**: 无需维护多个模型，训练时只需一次前向传播
4. **理论优雅**: 完整的数学推导证明了与 RLHF 的等价性

### 局限性
1. **Offline 性质**: 策略不主动探索—如果偏好数据覆盖差，性能会受限
2. **对数据质量敏感**: 偏好数据的噪声直接影响训练效果
3. **Bradley-Terry 假设**: 假设偏好是传递的、与提示无关的，现实中可能不成立
4. **缺乏探索**: 不同于 on-policy RL，无法收集新数据以改进弱点
5. **多轮对话受限**: 原始 DPO 被设计用于单轮评估，多轮场景需要修改

### 后续改进方向
- **Iterated DPO / Online DPO**: 策略更新后重新采样偏好数据
- **KTO**: 只需"好/坏"标签而非成对偏好
- **DPO 多轮扩展**: 如 Stepwise DPO、DPO for multi-turn

---

## 关联笔记

### 前置基础
- [[RLHF]] — DPO 试图简化的标准框架
- [[PPO]] — RLHF 中的标准 RL 算法

### 后续工作
- [[DeepSeekMath|GRPO]] — 结合组采样 + 隐式奖励
- [[RewardAnything]] — 泛化奖励模型的新方向

### 对比方法
- KTO — 无需成对数据的偏好优化
- SPIN — 自博弈偏好优化

---

## 速查卡片

> [!summary] DPO
> - **核心**: 语言模型即隐式奖励模型
> - **公式**: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})]$
> - **训练**: 1 阶段分类，无需奖励模型 + RL
> - **结果**: 摘要 Win Rate 61% > PPO 57%，对话任务唯一有效
> - **影响**: 开创免 RL 偏好优化新范式

---

*笔记创建时间: 2026-07-14 | 深度版*
