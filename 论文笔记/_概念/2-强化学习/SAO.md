---
type: concept
aliases: [Single-Rollout Asynchronous Optimization, 单次部署异步优化]
---

# SAO

## 定义
SAO (Single-Rollout Asynchronous Optimization) 是一种面向 LLM agentic post-training 的异步强化学习算法，核心是用单次 rollout 采样替代 GRPO 的组采样，配合双边重要性采样裁剪 (DIS) 和价值模型实现稳定的异步训练。

## 数学形式

**SAO 目标函数**:

$$J_{\text{SAO}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{rollout}}}\left[\frac{1}{|\tau|} \sum_{t=1}^{|\tau|} \text{mask}_t \cdot r_t(\theta) \cdot \hat{A}_t\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

**DIS Clipping**:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{rollout}}(a_t \mid s_t)}, \quad \text{mask}_t = \mathbf{1}\left[1 - \epsilon_\ell \le r_t(\theta) \le 1 + \epsilon_h\right]$$

## 核心要点
1. **单 rollout 替代组采样**: 每条 prompt 只生成 1 条 response，消除 GRPO 的 straggler 瓶颈
2. **DIS 双边裁剪**: 在 $[1-\epsilon_\ell, 1+\epsilon_h]$ 外的 token 完全排除出梯度计算
3. **价值模型回归**: 重新引入 $V_\phi$ 作为 baseline，配合高频更新 (K=2)、冻结 attention、Skip-Observation GAE
4. **异步原生设计**: 无需等待完整 batch，rollout 到达即消费

## 代表工作
- [[SAO]]: Hou et al., 2026. Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning. 达到 AIME2025 97.3%, 部署于 GLM-5.2 (750B-A40B).

## 相关概念
- [[GRPO]]: SAO 设计的主要对比和改进对象
- [[DIS]]: SAO 的核心稳定化机制
- [[Off-Policy]]: SAO 解决的异步 RL 核心问题
- [[Importance Sampling]]: DIS 的数学基础
- [[PPO]]: DIS clipping 的思想来源
- [[Skip-Observation GAE]]: SAO 的专用优势估计器
- [[Slime]]: SAO 所基于的开源训练框架
