---
title: "When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning"
method_name: "TACO"
authors: [Xiuyi Lou, Zicheng Xu, Yu-Neng Chuang, Hoang Anh Duy Le, Zhaozhuo Xu, Guanchu Wang, Vladimir Braverman]
year: 2026
venue: arXiv
tags: [reinforcement-learning, large-language-models, credit-assignment, GRPO, reasoning, RLVR, token-level-optimization]
zotero_collection: 
image_source: online
arxiv_html: https://arxiv.org/html/2607.07976v1
created: 2026-07-10
---

# 论文笔记：When Implausible Tokens Get Reinforced: Tail-Aware Credit Calibration for LLM Reinforcement Learning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Johns Hopkins University, Microsoft Research |
| 日期 | July 2026 |
| 项目主页 | https://github.com/xiuyilou/TACO |
| 对比基线 | [[GRPO]], [[DAPO]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.07976) / [Code](https://github.com/xiuyilou/TACO) |

---

## 一句话总结

提出 TACO 方法，通过局部上下文感知的尾风险评分校准 GRPO 的 uniform credit assignment，抑制不可靠尾 token 的正向更新。

---

## 核心贡献

1. **Positive-Credit Contamination 识别**: 发现 GRPO 风格 RLVR 中，低概率尾 token 获得与合理 token 相同的正向 credit，导致错误行为被强化
2. **Tail-Aware Credit CalibratiOn (TACO)**: 基于 token 采样概率和局部熵估计尾风险，软性地抑制高风险 token 的正向 credit
3. **全面实验验证**: 在 3 个 LLM 和 8 个基准上超越 GRPO 基线，且支持更稳定的长 horizon 训练

---

## 问题背景

### 要解决的问题
[[GRPO]] 风格的 [[RLVR]] 方法对所有 token 使用统一的序列级 advantage（uniform credit assignment），但同一正确轨迹中不同 token 的可靠性存在差异。

### 现有方法的局限
- 现有 token-level credit 方法需要额外推理或辅助模型，资源开销大
- 缺乏局部语义视角：不可靠尾 token 可能出现在高贡献或高熵位置，被现有方法误判为 informative token 而放大

### 本文的动机
利用生成时可以直接获取的统计量（token 概率、局部熵）作为代理，估计每个 token 属于不可靠尾分布的风险，在不增加显著计算成本的前提下抑制不当正向更新。

---

## 方法详解

### 模型架构

[[TACO]] 采用 **Credit Calibration** 架构：
- **输入**: 策略 $\pi_\theta$ 生成的 completion token 序列 $o_i = (o_{i,1}, \dots, o_{i,T_i})$
- **核心思想**: 利用局部生成统计量计算 token-level 尾风险评分，软性抑制高风险 token 的正向 advantage
- **核心模块**: [[Adaptive Tail-Risk Estimation]] 用于 [[Tail-Aware Credit Calibration]]
- **输出**: 校准后的 token-level advantage $\hat{A}^{\texttt{TACO}}_{i,t}$
- **额外开销**: 可忽略（仅使用前向传播中已有的概率和熵）

### 核心模块

#### 模块1: Adaptive Tail-Risk Estimation

**设计动机**: 利用 [[Token Probability]] 和 [[Entropy]] 区分"有用的稀有探索"和"不可靠的尾 token"

**具体实现**:
- 计算 token 的 surprisal $-\log p_{i,t}$
- 用局部熵 $H_{i,t}$ 作为上下文基准，衡量期望的 surprisal 水平
- 定义尾风险评分 $r^{\text{tail}}_{i,t} = -\log p_{i,t} - H_{i,t} + \log \alpha$
  - $r^{\text{tail}}_{i,t} > 0$: token 被视为有风险
  - $\alpha$: 控制尾风险识别严格度

#### 模块2: Tail-Aware Credit Calibration

**设计动机**: 基于尾风险评分软性下调高风险 token 的正向 credit，保留有用低概率模式的梯度

**具体实现**:
- 定义 token 权重 $w_{i,t} = \begin{cases} 1 - \lambda(1 - \exp(-r^{\text{tail}}_{i,t})), & r^{\text{tail}}_{i,t} > 0 \\ 1, & r^{\text{tail}}_{i,t} \leq 0 \end{cases}$
  - $\lambda \in (0,1)$: 最大抑制强度
- 校准后的 advantage: $\hat{A}^{\texttt{TACO}}_{i,t} = w_{i,t}^{\mathbb{I}[\hat{A}_i > 0]} \hat{A}_i$
  - 仅调制正向 advantage，负向保持不变

---

## 关键公式

### 公式1: [[GRPO]] 优化目标

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{q,\{o_i\}\sim\pi_{\theta_{\mathrm{old}}}}\left[\frac{1}{\sum_i T_i}\sum_{i=1}^G\sum_{t=1}^{T_i}\ell_{i,t}(\theta;\hat{A}_{i,t})\right]
$$

**含义**: GRPO 对一组采样 completion 中所有 token 的 PPO-style clipped surrogate 取平均，其中每个 token 共享相同的序列级 advantage $\hat{A}_{i,t} = \hat{A}_i$。

**符号说明**:
- $q$: 输入的 prompt
- $o_i$: 第 $i$ 个 completion，长度为 $T_i$
- $G$: group size（每组采样 completion 数量）
- $\hat{A}_i = (R_i - \mu)/\sigma$: 组归一化的序列级 advantage
- $\ell_{i,t}$: PPO-style clipped surrogate

### 公式2: [[Tail-Risk Score|尾风险评分]]

$$
r^{\mathrm{tail}}_{i,t} = \underbrace{-\log p_{i,t}}_{\text{token surprisal}} - \underbrace{H_{i,t}}_{\text{expected surprisal}} + \log \alpha
$$

**含义**: 衡量 token 超出局部熵预期的 surprisal 程度，正值表示该 token 处于不可靠尾分布中。

**符号说明**:
- $p_{i,t} = \pi_\theta(o_{i,t} \mid c_{i,t})$: 采样 token 概率
- $H_{i,t} = -\sum_{v \in \mathcal{V}} \pi_\theta(v \mid c_{i,t}) \log \pi_\theta(v \mid c_{i,t})$: 局部策略熵
- $\alpha$: 严格度超参数，$\alpha$ 越大识别越激进

### 公式3: [[Credit Suppression Weight|Credit 抑制权重]]

$$
w_{i,t} = \begin{cases}
1 - \lambda\left(1 - \exp\left(-r^{\mathrm{tail}}_{i,t}\right)\right), & r^{\mathrm{tail}}_{i,t} > 0, \\
1, & r^{\mathrm{tail}}_{i,t} \leq 0
\end{cases}
$$

**含义**: 高风险 token 的权重随尾风险评分增加而平滑衰减，下界为 $1-\lambda$；低风险 token 保持全权重。

**符号说明**:
- $\lambda \in (0,1)$: 最大抑制强度超参数
- $w_{i,t} \in [1-\lambda, 1]$: token 权重范围

### 公式4: [[TACO]] 校准 Advantage

$$
\hat{A}^{\texttt{TACO}}_{i,t} = w_{i,t}^{\mathbb{I}[\hat{A}_i > 0]} \hat{A}_i
$$

**含义**: 仅对正向 advantage 应用尾风险权重校准，负向 advantage 保持不变以保留失败轨迹的抑制信号。

**符号说明**:
- $\mathbb{I}[\hat{A}_i > 0]$: 指示函数，仅正向 advantage 被调制
- $\hat{A}_i$: 原始 GRPO 序列级 advantage

---

## 关键图表

### Figure 1: Overview / 方法概览

![Figure 1](https://arxiv.org/html/2607.07976v1/x1.png)

**说明**: TACO 在多个代表性基准和模型上持续优于 GRPO。

### Figure 2: Unreliable Tokens / 不可靠 token 示例

![Figure 2](https://arxiv.org/html/2607.07976v1/figures/12.png)

**说明**: 正确推理轨迹中出现的各种不可靠 token 实例，包括不必要的表格格式、损坏的数学公式、混合语言噪声等。

### Figure 3: Synthetic MDP / 合成 MDP 实验

![Figure 3a](https://arxiv.org/html/2607.07976v1/x2.png)
![Figure 3b](https://arxiv.org/html/2607.07976v1/x3.png)
![Figure 3c](https://arxiv.org/html/2607.07976v1/x4.png)

**说明**: 在合成序列 MDP 中验证 Positive-Credit Contamination 效应。(a) 轨迹越长差距越大；(b) 最优动作越稀疏差距越大；(c) group size 越小组差距越大。

### Figure 4: Training Dynamics / 训练动态

![Figure 4a](https://arxiv.org/html/2607.07976v1/x5.png)
![Figure 4b](https://arxiv.org/html/2607.07976v1/x6.png)
![Figure 4c](https://arxiv.org/html/2607.07976v1/x7.png)

**说明**: (a) TACO 训练准确率更高；(b) 策略熵更低更稳定；(c) 响应长度更长。

### Figure 5: Long-Horizon Training / 长 horizon 训练

![Figure 5a](https://arxiv.org/html/2607.07976v1/x8.png)
![Figure 5b](https://arxiv.org/html/2607.07976v1/x9.png)

**说明**: TACO 在 600 步扩展训练中持续提升，而 GRPO 在 450 步后 plateau 并退化。TACO 的熵曲线平滑稳定。

### Figure 6: Case Study / 案例研究

![Figure 6](https://arxiv.org/html/2607.07976v1/x10.png)

**说明**: TACO 选择性下调尾 token（红色标记，颜色越深抑制越强），同时保留合理推理步（蓝色标记）的完整 credit。

### Table 1: Main Results / 主要结果

| Method | AIME24 | AIME25 | AMC23 | MATH-500 | Minerva | Olympiad | MMLU-Pro | GPQA-D | Avg. |
|--------|--------|--------|--------|----------|---------|----------|----------|--------|------|
| **Qwen3-1.7B-Base** | | | | | | | | | |
| GRPO | 9.48 | 7.29 | 46.72 | 66.20 | 25.83 | 29.01 | 24.94 | 20.71 | 28.77 |
| GRPO w/ Adv. Reweighting | 11.25 | 8.44 | 47.03 | 64.75 | 24.63 | 29.11 | 29.84 | 23.36 | 29.80 |
| STAPO | 12.29 | 9.38 | 46.41 | 68.35 | 23.07 | 30.36 | 24.32 | 17.80 | 29.00 |
| **TACO (Ours)** | **14.38** | **9.06** | **49.45** | **68.35** | 25.74 | **31.71** | **30.45** | **24.43** | **31.70** |
| $$\Delta$$ vs GRPO | +4.90 | +1.77 | +2.73 | +2.15 | -0.09 | +2.70 | +5.51 | +3.72 | +2.93 |
| **Qwen3-4B-Base** | | | | | | | | | |
| GRPO | 25.73 | 20.83 | 68.13 | 76.50 | 34.65 | 36.35 | 36.26 | 26.14 | 40.57 |
| GRPO w/ Adv. Reweighting | 23.54 | 21.56 | 69.38 | 78.85 | 35.94 | 38.28 | 39.20 | 26.64 | 41.67 |
| STAPO | 24.69 | 21.98 | 72.89 | 77.00 | 33.64 | 36.83 | 38.84 | 25.88 | 41.47 |
| **TACO (Ours)** | **27.08** | **23.85** | 71.88 | **80.05** | 35.67 | **40.06** | **41.90** | **29.17** | **43.71** |
| $$\Delta$$ vs GRPO | +1.35 | +3.02 | +3.75 | +3.55 | +1.02 | +3.71 | +5.64 | +3.03 | +3.14 |
| **Qwen2.5-Math-7B** | | | | | | | | | |
| GRPO | 29.90 | 16.77 | 72.66 | 83.30 | 51.01 | 46.74 | 30.75 | 23.64 | 44.35 |
| GRPO w/ Adv. Reweighting | 28.96 | 17.29 | 74.84 | 82.95 | 50.18 | 46.77 | 29.05 | 24.43 | 44.31 |
| STAPO | 28.75 | 17.08 | 73.28 | 84.30 | 53.58 | 47.81 | 27.64 | 24.49 | 44.62 |
| **TACO (Ours)** | **32.40** | **19.79** | **78.44** | **84.65** | **55.51** | **49.41** | 30.53 | **24.84** | **46.95** |
| $$\Delta$$ vs GRPO | +2.50 | +3.02 | +5.78 | +1.35 | +4.50 | +2.67 | -0.22 | +1.20 | +2.60 |

**说明**: TACO 在所有 3 个模型和 8 个基准上平均性能最佳，在数学推理和 OOD 科学推理上均有一致提升。

### Table 2: Hyperparameter Sensitivity / 超参数敏感性

| $$\alpha$$ | $$\lambda$$ | AIME24 | AIME25 | AMC23 | MATH-500 | Minerva | Olympiad | MMLU-Pro | GPQA-D | Avg. |
|-----------|-----------|--------|--------|--------|----------|---------|----------|----------|--------|------|
| 0.01 | 0.6 | 15.10 | 9.58 | 47.50 | 68.15 | 24.82 | 28.97 | 30.95 | 23.99 | 31.13 |
| **0.01** | **0.9** | **14.38** | **9.06** | **49.45** | **68.35** | **25.74** | **31.71** | **30.45** | **24.43** | **31.70** |
| 0.005 | 0.6 | 12.81 | 8.64 | 48.15 | 67.20 | 24.72 | 29.38 | 28.85 | 26.26 | 30.75 |
| 0.005 | 0.9 | 13.23 | 9.06 | 46.57 | 67.55 | 25.55 | 30.08 | 29.85 | 25.81 | 30.96 |

**说明**: 默认设置 $(\alpha, \lambda) = (0.01, 0.9)$ 平均性能最佳，TACO 在合理超参数范围内保持有效。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DAPO-Math-17K | 17K | 数学推理训练数据 | 训练 |
| AIME 2024/2025 | - | 竞赛级数学 | 测试 |
| AMC 2023 | - | 竞赛级数学 | 测试 |
| MATH-500 | 500 | 多样化数学问题 | 测试 |
| Minerva Math | - | STEM 问题 | 测试 |
| OlympiadBench | - | 奥赛级问题 | 测试 |
| MMLU-Pro | - | 跨学科知识（OOD） | 测试 |
| GPQA-Diamond | - | 研究生级科学推理（OOD） | 测试 |

### 实现细节

- **Backbone**: Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen2.5-Math-7B
- **训练框架**: verl 框架，标准 GRPO recipe
- **训练数据**: DAPO-Math-17K
- **评估**: avg@32 (AIME, AMC), avg@4 (MATH-500, Minerva, Olympiad), avg@16 (MMLU-Pro, GPQA-Diamond)
- **基线**: GRPO (with clip-higher), GRPO w/ Adv. Reweighting, STAPO

### 关键发现

- TACO 在所有模型和基准上平均性能最优
- 在 OOD 科学推理基准（MMLU-Pro, GPQA-Diamond）上保持稳定提升
- 600 步扩展训练中 TACO 持续改进，GRPO 在 450 步后退化
- TACO 保持更稳定的策略熵，避免熵崩溃或爆炸

---

## 批判性思考

### 优点
1. **极低额外开销**: TACO 仅使用前向传播中已有的 token 概率和熵，无需额外模型或推理
2. **软性抑制设计**: 通过连续权重而非硬截断，保留有用低概率模式的梯度累积可能性
3. **自适应模型**: 尾风险评分基于每个策略自身的分布，自然适应不同模型的置信度特征

### 局限性
1. **仅处理正向 advantage**: 仅抑制正向 credit，未考虑负向 advantage 中可能存在的 false negative
2. **仅针对 Reward Signal**: 依赖 verifiable reward，在 reward 信号不可靠的场景（如偏好标注）中适用性未验证
3. **超参数依赖**: $\alpha$ 和 $\lambda$ 需要调优，过大 $\alpha$ 会抑制有用稀有 token

### 潜在改进方向
1. 扩展到负向 advantage 的双向校准
2. 在 RLHF/偏好优化场景中验证
3. 动态自适应 $\alpha$ 调度

### 可复现性评估
- [x] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[GRPO]]: 基础 RLVR 框架，TACO 在其上增加 token-level credit 校准
- [[PPO]]: GRPO 的底层算法原型

### 对比
- [[DAPO]]: GRPO 变体，关注解耦剪辑和动态采样
- [[STAPO]]: 梯度感知方法，抑制极端 token 更新
- [[GRPO w/ Adv. Reweighting]]: 重新加权低概率 token

### 方法相关
- [[Positive-Credit Contamination]]: 本文识别的核心问题
- [[Tail-Risk Score]]: TACO 的核心估计量
- [[Credit Assignment]]: 核心研究主题
- [[RLVR]]: 应用场景

### 硬件/数据相关
- [[DAPO-Math-17K]]: 训练数据集

---

## 速查卡片

> [!summary] When Implausible Tokens Get Reinforced: TACO
> - **核心**: 针对 GRPO 风格 RLVR 中不可靠尾 token 被错误强化的问题，提出基于局部上下文的 credit 校准方法
> - **方法**: 利用 token 概率和局部熵计算尾风险评分，软性抑制高风险 token 的正向 advantage
> - **结果**: 在 3 LLM x 8 基准上持续优于 GRPO，支持更稳定的长 horizon 训练
> - **代码**: https://github.com/xiuyilou/TACO

---

*笔记创建时间: 2026-07-10*
