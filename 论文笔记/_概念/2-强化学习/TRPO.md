---
name: trpo
description: Trust Region Policy Optimization — 信赖域策略优化
metadata:
  type: reference
---

# TRPO (Trust Region Policy Optimization)

## 一句话
通过**KL 散度约束**保证策略更新在信赖域内，[[PPO]] 的前身。

## 核心思想
$$\text{maximize}_\theta\ \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat{A}_t\right]$$
$$\text{s.t.}\ \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]] \leq \delta$$

## 与 PPO 对比
| 方面 | TRPO | [[PPO]] |
|------|:----:|:-------:|
| 约束方式 | 硬 KL 约束（二阶） | 软裁剪（一阶） |
| 实现 | 共轭梯度 + 二次近似 | SGD/Adam |
| 复杂度 | 高 | **低** |

## 关联
- [[PPO]]: 简化版 TRPO，更广使用
- Schulman et al., 2015: 原论文
