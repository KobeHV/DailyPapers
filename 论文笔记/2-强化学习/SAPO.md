---
title: "Soft Adaptive Policy Optimization"
method_name: "SAPO"
authors: [Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin]
year: 2025
venue: arXiv
tags: [reinforcement-learning, policy-optimization, soft-clipping, moe, qwen, variance-reduction]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2511.20347v1
created: 2026-07-14
updated: 2026-07-14
aliases: [SAPO, Soft Adaptive Policy Optimization]
---

# 论文笔记：Soft Adaptive Policy Optimization (SAPO)

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | Qwen Team, Alibaba Inc. |
| **作者** | Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin |
| **发表** | arXiv:2511.20347, 2025年12月 |
| **对比基线** | [[GSPO]], [[DeepSeekMath|GRPO]], GRPO-R2 |
| **链接** | [arXiv](https://arxiv.org/abs/2511.20347) |

---

## 一句话总结

> 用 **sigmoid 软门控**替代 GSPO/GRPO 的硬裁剪，让梯度随重要性比偏离程度**平滑衰减**而非直接归零，MoE 训练稳定且无需 Routing Replay。

---

## 问题背景

### 核心问题：Token 级 Importance Ratio 的高方差

Token 级 importance ratio $r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t})}{\pi_{\theta_{\text{old}}}(y_{i,t})}$ 表现出**高方差**：

- 在 MoE 模型中因路由异质性**进一步放大**
- 长响应中极端 ratio 值频繁出现
- 导致训练不稳定

### 现有方案的局限

**硬裁剪方案**（GRPO、GSPO）的问题：

```
GRPO: min(r·A, clip(r, 1-ε, 1+ε)·A)
GSPO: min(s·A, clip(s, 1-ε, 1+ε)·A)
```

- **全有或全无**: ratio 在区间内完全保留，超出完全丢弃
- **浪费学习信号**: 接近边界但仍有用的信号被截断
- **GSPO 的序列级级联问题**: 一个偏离 token 可导致整个序列被裁剪

### 本文的动机

设计一个**平滑的信任域**—梯度随偏离程度**连续衰减**而非二值化截断。

---

## 方法详解

### SAPO 目标函数

$$\mathcal{J}_{\text{SAPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \underbrace{f_{i,t}(r_{i,t}(\theta))}_{\text{软门控}}\cdot\hat{A}_{i,t}\right]$$

### 软门控函数

$$f_{i,t}(x) = \sigma\left(\tau_{i,t} \cdot (x-1)\right) \cdot \frac{4}{\tau_{i,t}}$$

其中：
- $\sigma$: sigmoid 函数
- $\tau_{i,t}$: 温度参数，控制衰减速度
- $x = r_{i,t}(\theta)$: token 级 importance ratio

**非对称温度**:
$$\tau_{i,t} = \begin{cases}\tau_{pos} = 1.0 & \hat{A}_{i,t} > 0 \\ \tau_{neg} = 1.05 & \hat{A}_{i,t} < 0\end{cases}$$

### 门控函数的关键性质

**1. 在 r=1 处归一化**:

$$f_{i,t}(1) = \sigma(0) \cdot \frac{4}{\tau} = 0.5 \cdot \frac{4}{\tau} = \frac{2}{\tau}$$

乘以 $\frac{\tau}{2}$ 的缩放因子后可保证 $f(1)=1$。

**2. 平滑衰减**:

当 $r$ 偏离 1 时，$f(r)$ 指数级衰减而非瞬间归零：

| $r$ (偏离程度) | GRPO 权重 | SAPO 权重 ($\tau=1$) |
|:-------------:|:---------:|:-------------------:|
| 1.0 (在策略) | 1 | 1 |
| 1.2 (中等偏离) | 1 | **~0.55** |
| 1.5 (大偏离) | 1 (未裁剪) 或 0 (已裁剪) | **~0.18** |
| 2.0 (严重偏离) | 0 (已裁剪) | **~0.04** |

**3. 梯度形式**:

$$\nabla_\theta \mathcal{J}_{\text{SAPO}} = \mathbb{E}\left[\frac{1}{G}\sum_i\frac{1}{|y_i|}\sum_t \underbrace{w_{i,t}(\theta)}_{\text{tu} \text{平滑权重}}\cdot r_{i,t}(\theta)\cdot\nabla\log\pi_\theta(y_{i,t})\cdot\hat{A}_i\right]$$

其中：
- $w_{i,t}(\theta) = 4 \cdot p_{i,t}(\theta) \cdot (1 - p_{i,t}(\theta))$
- $p_{i,t}(\theta) = \sigma(\tau_{i,t} \cdot (r_{i,t}(\theta) - 1))$

$w$ 在 $r=1$ 时取最大值 1，偏离时平滑衰减到 0。

### 序列级连贯性

在温和假设下，SAPO 的 token 级门控可聚合成序列级门控：

$$g(\log s_i(\theta)) = \text{sech}^2\left(\frac{\tau_i}{2} \cdot \log s_i(\theta)\right)$$

**近似误差有界**:

$$D_i(\theta) \leq \frac{\tau_i^2}{4} \cdot \text{Var}_i(\theta)$$

其中 $\text{Var}_i(\theta)$ 是序列内 token 级 ratio 的方差。MoE 模型的 $\text{Var}_i(\theta)$ 显著更大，因此 token 级处理更有必要。

### 非对称温度的动机

关键洞察来自梯度在词汇表上的传播分析：

对于采样 token $v = y_{i,t}$:

$$\frac{\partial \mathcal{L}}{\partial \logit_{v}} = \underbrace{(1 - \pi_\theta(v|\cdots))}_{\text{提高该 token}} \cdot \hat{A}_{i,t}$$

对于未采样 token $u \neq y_{i,t}$:

$$\frac{\partial \mathcal{L}}{\partial \logit_{u}} = \underbrace{(-\pi_\theta(u|\cdots))}_{\text{降低其他 token}} \cdot \hat{A}_{i,t}$$

**当 $\hat{A} < 0$**：需要降低采样 token 的概率，同时**提高大量未采样 token** 的概率。词汇表规模通常为 100K+，负梯度扩散到太多无关 token → **严重不稳定**。

**解决方案**: $\tau_{neg} > \tau_{pos}$ — 负优势时梯度衰减更快。

### 消融验证

| $\tau_{neg} : \tau_{pos}$ | 训练稳定性 |
|:-------------------------:|:----------:|
| 1.05 : 1.0 | **最稳定** ✅ |
| 1.0 : 1.0 | 中等稳定性 |
| 0.95 : 1.0 | **显著不稳定** ❌ |

### GRPO vs GSPO vs SAPO 梯度对比

```
GRPO 梯度:
  ∇J = E[ (1/G) Σ_i A_i · (1/|y_i|) Σ_t min(r_t, clip(r_t)) · ∇log π(y_t) ]

GSPO 梯度:
  ∇J = E[ (1/G) Σ_i s_i · A_i · (1/|y_i|) Σ_t ∇log π(y_t) ]
         ↑ 序列级统一权重

SAPO 梯度:
  ∇J = E[ (1/G) Σ_i (1/|y_i|) Σ_t f(r_t)·A_i · ∇log π(y_t) ]
         ↑ token 级平滑权重
```

---

## GSPO vs SAPO 详细对比

| 维度 | GSPO | SAPO |
|------|:----:|:----:|
| **核心思想** | 序列级 IS ratio | 软门控函数 |
| **IS 层面** | 序列级 | Token 级（保留细粒度） |
| **裁剪方式** | min/clip 硬边界的裁剪 | sigmoid **平滑**衰减 |
| **序列一致性** | 强（整个序列等权） | 中等（可聚合成序列级） |
| **Token 适应性** | 无 | **每个 token 独立权重** |
| **非对称处理** | 无 | **有**（$\tau_{neg}>\tau_{pos}$） |
| **MoE Routing Replay** | 不需要 | 不需要 |
| **梯度信号** | 统一 | **平滑/连续** |

---

## 实验结果

### 数学推理 (Qwen3-30B-A3B)

**冷启动训练** ($\tau_{pos}=1.0, \tau_{neg}=1.05$):

| 方法 | AIME25 | HMMT25 | BeyondAIME |
|:----|:------:|:------:|:----------:|
| GRPO-R2 | 基线 | 基线 | 基线 |
| GSPO | 基线+ | 基线+ | 基线+ |
| **SAPO** | **最优** | **最优** | **最优** |

**关键发现**:
- GSPO 和 GRPO-R2 出现**早期训练崩溃**
- SAPO **全程稳定**且达到 higher final performance
- SAPO 不需要 Routing Replay

### Qwen3-VL 多模态训练

| 基准 | GSPO | GRPO-R2 | **SAPO** |
|:----|:----:|:-------:|:--------:|
| AIME25 | 基线 | 基线 | **最优** |
| LiveCodeBench v6 | 基线 | 基线 | **最优** |
| ZebraLogic | 基线 | 基线 | **最优** |
| MathVision | 基线 | 基线 | **最优** |

> SAPO 在不同模型大小和架构（MoE 和 Dense）上**一致提升**。

---

## 批判性分析

### 优点
1. **数学优雅**: 软门控的梯度形式 $w = 4p(1-p)$ 是对 sigmoid 的自然推导
2. **非对称温度有理论依据**: 基于词汇表梯度扩散分析，非凭经验
3. **MoE 友好**: 不需要 Routing Replay，简化工程
4. **保留细粒度**: 每个 token 有独立权重，不牺牲 token 级信息

### 局限性
1. **额外超参数**: 引入 $\tau_{pos}, \tau_{neg}$ 两个温度参数
2. **计算开销**: sigmoid 计算比 min/clip 稍高（通常可忽略）
3. **温度调参**: $\tau_{neg}=1.05$ 在不同模型上可能需要调整
4. **仅在 Qwen 模型验证**: 可迁移性需更多实验

---

## 关联笔记

### 同系列（阿里 Qwen 团队）
- [[GSPO]] — 序列级 IS ratio（同团队同期工作）
- [[StabilizingRL]] — 形式化框架（同团队）
- [[DeepSeekMath|GRPO]] — 共同改进起点

### 对比
- [[SAO]] — 不同问题：异步 RL 兼容性

---

## 速查卡片

> [!summary] SAPO
> - **核心**: 软门控替代硬裁剪
> - **公式**: $f(x) = \sigma(\tau(x-1))·4/\tau$，$\tau_{neg}>\tau_{pos}$
> - **优势**: MoE 无需 Routing Replay，梯度平滑衰减，训练稳定
> - **结果**: Qwen3-VL 多模态训练一致提升
> - **影响**: 首次提出非对称温度处理负优势梯度扩散

---

*笔记创建时间: 2026-07-14 | 深度版*
