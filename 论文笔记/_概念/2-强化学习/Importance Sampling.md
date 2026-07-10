---
type: concept
aliases: [重要性采样, IS, importance weighting]
---

# Importance Sampling

## 定义
重要性采样 (Importance Sampling) 是一种统计技术，用于在数据来自分布 $\mu$ 而非目标分布 $\pi$ 时，通过加权修正来无偏估计 $\pi$ 下的期望。

## 数学形式

基本形式:

$$\mathbb{E}_{x \sim \pi}[f(x)] = \mathbb{E}_{x \sim \mu}\left[\frac{\pi(x)}{\mu(x)} f(x)\right]$$

在 RL 中的 token 级形式:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\mu(a_t \mid s_t)}$$

$$\hat{A}_t^{\text{corrected}} = r_t(\theta) \cdot \hat{A}_t$$

序列级乘积:

$$\frac{\pi_\theta(\tau)}{\mu(\tau)} = \prod_{t=1}^{T} \frac{\pi_\theta(a_t \mid s_t)}{\mu(a_t \mid s_t)} = \prod_{t=1}^{T} r_t(\theta)$$

## 核心要点
1. **无偏但高方差**: 理论上提供无偏估计，但在长序列上 ratio 乘积的方差指数增长
2. **Clipping 的必要性**: PPO 通过 $\min(r_t A, \text{clip}(r_t)A)$ 控制方差；SAO 的 DIS 通过双边 mask 更进一步
3. **在 LLM RL 中**: importance ratio 在 token 级别计算，每个 token 的 $r_t$ 衡量该 token 在当前策略下的相对概率变化
4. **Per-token vs Per-sequence**: 现代 LLM RL 通常使用 per-token IS（不累积乘积），大幅降低方差

## 代表工作
- [[PPO]]: 引入 clipped importance sampling 控制 off-policy 更新
- [[SAO]]: 提出 DIS，使用 rollout policy 作为直接分母 + 双边严格屏蔽
- [[GRPO]]: 使用 per-token importance ratio with PPO-style clipping
- [[IMPALA]]: 使用 V-trace 做 off-policy 重要性采样修正

## 相关概念
- [[Off-Policy]]: IS 是处理 off-policy 数据的核心工具
- [[DIS]]: SAO 中的双边重要性采样裁剪机制
- [[PPO]]: 标准单边 IS clipping
- [[Trust Region]]: IS clipping 的约束理论基础
- [[GAE]]: 与 IS 结合使用的优势估计方法
