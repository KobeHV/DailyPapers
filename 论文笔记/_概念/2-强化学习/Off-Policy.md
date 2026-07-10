---
type: concept
aliases: [off-policy learning, 离策略学习, policy lag]
---

# Off-Policy

## 定义
Off-Policy (离策略学习) 指训练数据由与当前策略不同的行为策略生成的情况。在异步 RL 中，由于 rollout 生成和模型更新之间存在时间差，行为策略 (rollout policy) 通常滞后于当前训练策略，产生 policy lag 和 off-policy effects。

## 数学形式

On-policy 训练假设数据来自当前策略:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

Off-policy 训练的数据来自行为策略 $\mu \neq \pi_\theta$，需要通过 importance sampling 修正:

$$J(\theta) = \mathbb{E}_{\tau \sim \mu}\left[\frac{\pi_\theta(\tau)}{\mu(\tau)} R(\tau)\right] = \mathbb{E}_{\tau \sim \mu}\left[\prod_{t} \frac{\pi_\theta(a_t|s_t)}{\mu(a_t|s_t)} R(\tau)\right]$$

在 token 级别:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\mu(a_t \mid s_t)}$$

## 核心要点
1. **异步 RL 的根本挑战**: rollout 由旧版本模型生成，当前模型已经更新了若干步，导致数据"过时"
2. **方差问题**: importance ratio 乘积在长序列上方差极大，需要 clipping/regularization 控制
3. **Policy Lag**: 异步训练中 rollout 策略版本与训练策略版本之间的步数差，lag 越大 off-policy 程度越严重
4. **解决思路**: (a) 减小 lag (单 rollout), (b) 控制 ratio 范围 (clipping/masking), (c) 使用价值模型减少方差

## 代表工作
- [[PPO]]: 通过 clipping 将 off-policy 更新约束在 trust region 内
- [[SAO]]: 通过单 rollout 最小化 lag + DIS 双边 mask 控制 off-policy 效应
- [[VCPO]]: 使用 ESS 引导步长缩放 + closed-form off-policy optimal baseline

## 相关概念
- [[Importance Sampling]]: off-policy 修正的数学工具
- [[DIS]]: SAO 中控制 off-policy 效应的 clipping 机制
- [[SAO]]: 在异步 training 中系统解决 off-policy 问题的方法
- [[PPO]]: 最早的 clipped off-policy 方法
- [[Trust Region]]: off-policy 更新的约束框架
