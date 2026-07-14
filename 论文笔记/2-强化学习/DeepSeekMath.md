---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
method_name: "DeepSeekMath"
authors: [Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo]
year: 2024
venue: arXiv
tags: [mathematical-reasoning, grpo, reinforcement-learning, data-curation, open-model]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2402.03300v1
created: 2026-07-14
---

# 论文笔记：DeepSeekMath — GRPO

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | DeepSeek-AI / 清华大学 / 北京大学 |
| 日期 | April 2024 |
| 对比基线 | Gemini-Ultra, GPT-4, Minerva |
| 链接 | [arXiv](https://arxiv.org/abs/2402.03300) |

---

## 一句话总结

> 提出 GRPO（无 critic 的 PPO 变体）和 DeepSeekMath Corpus（120B 数学 token），7B 模型在 MATH 上达 51.7%，逼近 GPT-4。

---

## 核心贡献

1. **DeepSeekMath Corpus**: 从 Common Crawl 经 4 轮迭代提取的 120B 高质量数学 token
2. **GRPO (Group Relative Policy Optimization)**: 去掉价值网络，用组内奖励归一化替代 critic

---

## 方法详解

### GRPO 目标函数

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\left\{\min\left(\frac{\pi_\theta(o_{i,t})}{\pi_{\theta_{\text{old}}}(o_{i,t})}\hat{A}_{i,t},\ \text{clip}(\cdots)\right)
- \beta\mathbb{D}_{KL}[\pi_\theta||\pi_{\text{ref}}]\right\}\right]$$

**组内优势**（无价值网络）:
$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

**KL 散度估计**（无偏）:
$$\mathbb{D}_{KL}[\pi_\theta||\pi_{\text{ref}}] = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

### PPO vs GRPO

| 方面 | PPO | GRPO |
|------|:---:|:----:|
| 价值网络 | 需要（双倍显存） | **不需要** |
| 优势估计 | GAE（需价值函数） | **组内归一化** |
| KL 项 | 加入奖励 | **直接加入损失** |
| 内存 | 高 | **低** |

---

## 实验结果

| 模型 | GSM8K | MATH |
|------|:-----:|:----:|
| DeepSeekMath-Base 7B | 64.2% | 36.2% |
| DeepSeekMath-Instruct 7B | 82.9% | 46.8% |
| **DeepSeekMath-RL 7B (GRPO)** | **88.2%** | **51.7%** |
| + Self-Consistency (64) | - | **60.9%** |

**RL 为何有效？** 提升的是 **Maj@K** 而非 **Pass@K** → 使输出分布更稳健，非增强基础能力。

---

## 关联笔记

### 前置
- [[PPO]]: GRPO 的基础算法

### 后续工作
- [[GSPO]]: 序列级 importance ratio 改进 GRPO
- [[SAPO]]: 软门控替代硬裁剪
- [[SAO]]: 异步单 rollout 优化
- [[StabilizingRL]]: 形式化分析 GRPO 的问题

---

## 速查卡片

> [!summary] DeepSeekMath / GRPO
> - **核心**: 无 Critic 的组相对策略优化
> - **方法**: 组内奖励归一化 + 免价值网络
> - **结果**: DeepSeekMath 7B MATH 51.7%, 逼近 GPT-4
> - **影响**: GRPO 成为 LLM RL 后训练的主流算法

---

*笔记创建时间: 2026-07-14*
