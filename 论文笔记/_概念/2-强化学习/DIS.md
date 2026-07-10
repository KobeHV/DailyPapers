---
type: concept
aliases: [Direct Double-Sided Importance Sampling, 直接双边重要性采样]
---

# DIS

## 定义
DIS (Direct Double-Sided Importance Sampling) 是 SAO 中的稳定化机制，使用严格的双边 token 级重要性采样裁剪，将 importance ratio 超出预设区间的 token 从梯度计算中完全屏蔽。

## 数学形式

**Importance Ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{rollout}}(a_t \mid s_t)}$$

其中 $\pi_{\text{rollout}}$ 是实际生成 rollout 时的策略（直接使用 rollout engine 记录的 log-probability，无需存储历史 checkpoint）。

**Double-Sided Token Mask**:

$$\text{mask}_t = \mathbf{1}\left[1 - \epsilon_\ell \le r_t(\theta) \le 1 + \epsilon_h\right]$$

**Effective Loss**:

$$L_{\text{DIS}}(\theta) = -\frac{1}{\sum_t \text{mask}_t} \sum_t \text{mask}_t \cdot r_t(\theta) \cdot \hat{A}_t$$

被 mask 的 token 对梯度贡献为零。

## 核心要点
1. **双边严格**: 与 PPO 的单边 clip 不同，DIS 在 $r_t$ 过高或过低时均屏蔽 —— 对两个方向的 policy divergence 一视同仁
2. **Token 级粒度**: 每个 token 独立判断，允许局部的 on-policy 对齐
3. **直接行为代理**: 使用 rollout policy 的 log-probability 作为分母，无需维护 $\pi_{\text{old}}$
4. **屏蔽而非裁剪**: 超出区间的 token 完全排除（而不是 clip 到一个边界值），更严格地防止 off-policy 更新

## 代表工作
- [[SAO]]: Hou et al., 2026. 首次提出 DIS，用于异步 agentic RL 的稳定化训练。

## 相关概念
- [[Importance Sampling]]: DIS 的数学基础
- [[Off-Policy]]: DIS 解决的核心问题（policy lag）
- [[PPO]]: DIS 从中继承了 clipping 思想但做了双边严格化
- [[SAO]]: DIS 的提出和部署论文
- [[GRPO]]: 其单边 clip 在异步场景下不稳定，DIS 提供替代方案
