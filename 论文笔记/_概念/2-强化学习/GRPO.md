---
type: concept
aliases: [Group Relative Policy Optimization, 组相对策略优化]
---

# GRPO

## 定义
GRPO (Group Relative Policy Optimization) 是一种无需价值模型的强化学习算法，通过为每条 prompt 采样多条 response 并在组内计算标准化相对优势来优化策略。被 DeepSeek-R1、Qwen3、GLM-5 等主流 LLM 广泛使用。

## 数学形式

**组内标准化优势**:

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

其中 $G$ 是 group size（每条 prompt 的 response 数）。

**GRPO 目标函数**:

$$J_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(r_{i,t}(\theta) A_i,\; \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) A_i\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

**Token 级 importance ratio**:

$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$$

## 核心要点
1. **无需价值模型**: 用组内 reward 标准化替代 critic，减少显存和计算开销
2. **组采样必需**: $G \ge 2$ 才能计算有效的相对优势，单 rollout 下优势退化为零
3. **同步训练设计**: 需要等待完整 group 才能更新模型，不适合异步 agentic 训练
4. **三代模型**: 相比 PPO（4 模型: actor + critic + ref + reward），GRPO 仅需 3 模型

## 代表工作
- [[GRPO]]: Shao et al., 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. 首次提出 GRPO.
- [[SAO]]: Hou et al., 2026. 针对 GRPO 在异步设置下的局限，提出单 rollout 替代方案.

## 相关概念
- [[SAO]]: 替代 GRPO 的异步单 rollout 方法
- [[PPO]]: GRPO 的算法基础（clipping mechanism）
- [[DAPO]]: GRPO 的变体（非对称 clipping + 动态采样）
- [[Importance Sampling]]: GRPO 使用的 token 级 IS
- [[Off-Policy]]: GRPO 在异步场景下遇到的核心问题
- [[DIS]]: SAO 中替代 GRPO clipping 的双边机制
