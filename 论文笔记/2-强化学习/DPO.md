---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
method_name: "DPO"
authors: [Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn]
year: 2024
venue: arXiv / NeurIPS 2024
tags: [preference-optimization, rlhf, alignment, reward-model-free, classification]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2305.18290v1
created: 2026-07-14
---

# 论文笔记：Direct Preference Optimization

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Stanford University |
| 日期 | July 2024 |
| 对比基线 | [[PPO]]-based RLHF, Preferred-FT, Unlikelihood |
| 链接 | [arXiv](https://arxiv.org/abs/2305.18290) |

---

## 一句话总结

> 通过变量变换直接优化偏好，绕过显式奖励模型和 RL 训练，将 RLHF 三阶段简化为一阶段分类问题。

---

## 核心贡献

1. **隐式奖励函数**: 证明语言模型本身就是隐式奖励模型，消除独立奖励模型的需要
2. **DPO 损失函数**: 简单的二元交叉熵损失，直接优化偏好数据
3. **理论等价性**: 证明 DPO 与 RLHF 优化同一目标函数

---

## 方法详解

### 核心洞察

RLHF 最优策略的闭式解:
$$\pi_r(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

反解出奖励函数:
$$r(x,y) = \beta\log\frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)$$

代入 Bradley-Terry 偏好模型 → $Z(x)$ 消掉:
$$p^*(y_1 \succ y_2 | x) = \frac{1}{1 + \exp\left(\beta\log\frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} - \beta\log\frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}\right)}$$

### DPO 损失函数

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**梯度**:
$$\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\mathbb{E}\left[\underbrace{\sigma(\hat{r}_\theta(y_l)-\hat{r}_\theta(y_w))}_{\text{排名错误越严重 → 权重越大}}\left(\underbrace{\nabla_\theta\log\pi(y_w)}_{\uparrow y_w} - \underbrace{\nabla_\theta\log\pi(y_l)}_{\downarrow y_l}\right)\right]$$

---

## 实验结果

| 任务 | DPO | PPO-based RLHF |
|-----|:---:|:--------------:|
| TL;DR 摘要 Win Rate | **~61%** | ~57% |
| 单轮对话 | **唯一有效** | 未能超越基线 |
| 训练阶段 | **1 阶段** | 3 阶段 |
| 训练采样 | **不需要** | 需要 (on-policy) |
| 稳定性 | **稳定** | 高方差 |

---

## 对比: RLHF vs DPO

| 方面 | RLHF (PPO) | DPO |
|------|:----------:|:---:|
| 阶段数 | SFT → RM → PPO (3) | SFT → DPO (**1**) |
| 奖励模型 | 显式训练 | **隐式**由策略表示 |
| 训练时采样 | 需要 (on-policy) | **不需要** (offline) |
| 计算成本 | 高 | **低** |
| 理论目标 | $\max \mathbb{E}[r] - \beta\text{KL}$ | 相同目标，不同优化方式 |

---

## 关联笔记

### 基于
- [[PPO]]: RLHF 中的标准 RL 算法
- [[RLHF]]: DPO 试图简化的框架

### 后续工作
- [[DeepSeekMath|GRPO]]: 结合组采样的策略优化
- [[StabilizingRL]]: 对 RL 目标的更深入形式化

---

## 速查卡片

> [!summary] DPO
> - **核心**: 语言模型即隐式奖励模型
> - **方法**: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})]$
> - **结果**: 摘要 Win Rate 61%，超越 PPO-based RLHF
> - **影响**: 开创免 RL 偏好优化范式，引用量 5000+

---

*笔记创建时间: 2026-07-14*
