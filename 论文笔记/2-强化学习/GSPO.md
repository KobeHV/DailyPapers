---
title: "Group Sequence Policy Optimization"
method_name: "GSPO"
authors: [Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin]
year: 2025
venue: arXiv
tags: [reinforcement-learning, grpo, sequence-level, importance-sampling, moe]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2507.18071v2
created: 2026-07-14
---

# 论文笔记：Group Sequence Policy Optimization (GSPO)

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Qwen Team, Alibaba Inc. |
| 日期 | July 2025 |
| 对比基线 | [[DeepSeekMath|GRPO]], [[PPO]] |
| 链接 | [arXiv](https://arxiv.org/abs/2507.18071) |

---

## 一句话总结

> GRPO 的 token 级重要性采样从根本上就是** ill-posed** 的—提出**序列级** importance ratio，Qwen3 的基础算法。

---

## 核心贡献

1. **序列级 Importance Ratio**: 用序列几何均值代替 token 级重要性比，对齐重要性采样原理
2. **消除 Routing Replay**: 对 MoE 模型无需 Routing Replay，简化基础设施
3. **大幅提升稳定性**: 解决 GRPO 在长响应任务中的不可逆崩溃问题

---

## 方法详解

### GRPO 的根本问题

Token 级 IS 权重 $\frac{\pi_\theta(y_t)}{\pi_{\theta_{\text{old}}}(y_t)}$ 有问题：

1. 每个 next-token 分布只有 **1 个样本** → 无法做分布校正
2. 长序列中噪声累积
3. 裁剪放大了而非缓解了问题
4. 一旦崩溃**不可逆**

### 序列级 Importance Ratio

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t})}{\pi_{\theta_{\text{old}}}(y_{i,t})}\right)$$

**GSPO 目标**:
$$\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i \min\left(s_i(\theta)\cdot A_i,\ \text{clip}(s_i,1-\epsilon,1+\epsilon)\cdot A_i\right)\right]$$

### GRPO vs GSPO 对比

| 方面 | GRPO (token 级) | GSPO (序列级) |
|------|:--------------:|:-------------:|
| Importance ratio | 每个 token 独立 | **整个序列一个** |
| 梯度权重 | token 间不等 → 高方差 | **所有 token 等权重** |
| 裁剪比例 | 0.13% | **15%**（但更高效） |
| MoE 训练 | 需要 Routing Replay | **不需要** |
| 训练稳定性 | 可能不可逆崩溃 | **全程稳定** |
| 基础设施 | 需训练引擎重算 token 似然 | **可直接用推理引擎** |

---

## 关键结果

- Qwen3-30B-A3B 上 GSPO **持续优于 GRPO**
- 在相同训练计算量下，训练准确率和基准性能都更好
- **裁剪了更多 token（15% vs 0.13%）却更高效**→ 证明 GRPO 的 token 级梯度估计本质上有噪声

---

## 关联笔记

### 基于
- [[DeepSeekMath|GRPO]]: GSPO 改进的基础算法
- [[PPO]]: 策略梯度框架

### 同系列
- [[SAPO]]: 同时期阿里的另一 GRPO 改进（软门控）
- [[StabilizingRL]]: 形式化框架
- [[SAO]]: 异步单 rollout 优化

---

## 速查卡片

> [!summary] GSPO
> - **核心**: 序列级 importance ratio 替代 token 级
> - **方法**: $s_i = (\pi_\theta(y_i)/\pi_{\theta_{\text{old}}}(y_i))^{1/|y_i|}$
> - **结果**: 稳定训练 MoE，无需 Routing Replay，Qwen3 基础算法
> - **影响**: 从根本原理上解决了 GRPO 的不稳定性

---

*笔记创建时间: 2026-07-14*
