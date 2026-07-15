---
title: "Multi-module GRPO: Composing Policy Gradients and Prompt Optimization for Language Model Programs"
method_name: "mmGRPO"
authors: [Noah Ziems, Dilara Soylu, Lakshya A Agrawal, Isaac Miller, Liheng Lai, Chen Qian, Kaiqiang Song, Meng Jiang, Dan Klein, Matei Zaharia, Karel D'Oosterlinck, Christopher Potts, Omar Khattab]
year: 2025
venue: ACM CAIS 2026
tags: [rl, grpo, prompt-optimization, multi-module, lm-programs, dspy]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2508.04660v2
created: 2026-07-15
---

# 论文笔记：Multi-module GRPO: Composing Policy Gradients and Prompt Optimization for Language Model Programs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Notre Dame, Stanford University, UC Berkeley, MIT, Databricks, Contextual AI |
| 日期 | August 2025 (v1) / May 2026 (v2) |
| 项目主页 | https://dspy.ai |
| 对比基线 | [[GRPO]], [[PPO]], MIPROv2 |
| 链接 | [arXiv](https://arxiv.org/abs/2508.04660) / [Code (DSPy)](https://github.com/stanfordnlp/dspy) / [ACM](https://dl.acm.org/doi/10.1145/3786335.3813164) |

---

## 一句话总结

> 将 GRPO 扩展到多模块 LM 程序，提出模块级分组策略梯度，并与提示优化组合实现 11% 的准确率提升。

---

## 核心贡献

1. **mmGRPO**: 将 [[GRPO]] 推广到多模块 LM 程序，通过模块级轨迹分组解决变长/异构结构的策略梯度优化问题
2. **模块级分组策略**: 按模块标识符和调用顺序对齐不同轨迹中的同模块调用，独立计算每组优势并更新对应模块权重
3. **BetterTogether 框架**: 将 [[提示优化]]（MIPROv2）与 mmGRPO 在线 RL 组合，先优化提示模板再微调模型权重

---

## 问题背景

### 要解决的问题

现代 LM 应用通常是由多个 LM 调用、不同提示模板和控制流逻辑组成的**模块化程序**（如多跳检索、隐私保护委派、分类管线）。现有的 [[GRPO]] 仅支持单模块设置（一次 rollout 一次 LM 调用），无法直接用于多模块程序。

### 现有方法的局限

- 标准 GRPO 要求组内所有 rollout 共享相同的输入 prompt，而多模块程序中各模块的输入因轨迹不同而变化
- 多模块轨迹可能具有**可变长度和异构结构**（控制流分支、模块被调用不同次数、解析失败提前终止）
- 简单地将 GRPO 应用于完整轨迹（轨迹级分组）无法区分不同模块的贡献

### 本文的动机

作者提出在**模块级别**而非轨迹级别进行 GRPO 分组：将不同轨迹中同一模块的调用对齐，独立进行组内优势归一化和策略更新。同时与 [[提示优化]] 组合，实现提示模板和模型权重的联合优化（BetterTogether）。

---

## 方法详解

### 整体架构

mmGRPO 采用 **多模块策略梯度** 架构：
- **输入**: 程序级输入 $x$
- **采样过程**: 采样 $G$ 条完整轨迹 $(y, \rho) \sim \Phi(x)$，每条轨迹包含模块调用序列 $\rho = [\zeta_1, \zeta_2, \dots, \zeta_{|\rho|}]$，每个轨迹步 $\zeta_t = \langle M_t, q_t, o_t \rangle$ 记录模块标识、输入和输出
- **核心模块**: [[GRPO]] 分组器（FormModuleLevelGroups）+ 模块级策略更新
- **Reward**: 程序级标量奖励 $r = \mu(y, \rho, m)$，**所有模块共享同一奖励**（[[Credit Assignment|统一信用分配]]）
- **总参数**: Llama 3.1-8B-Instruct / Qwen3-8B（LoRA 微调）

### 核心模块

#### 模块1: FormModuleLevelGroups（模块级分组器）

**设计动机**: 不同轨迹中的同一模块调用在结构上具有可比性，但输入各异。将同一模块的调用分组可以对齐后使用 [[GRPO]] 的组内优势归一化。

**具体实现**:
- 按 `(module_id, relative_index)` 为键对轨迹中的模块调用进行分组
- 使用 `PAD_GROUPS` 处理变长轨迹（提供 `fill` 复制填充和 `truncate` 截断两种模式）
- 使用 `SELECTK_DIVERSE_ELEMENTS` 选择最大化奖励方差的轨迹子集，改善泛化

#### 模块2: 模块级策略更新

**设计动机**: 每组模块调用的策略梯度方向不同，应独立更新对应模块的 LM 权重。

**具体实现**:
- 对每个模块组 $\{(q_i, o_i, r_i)\}_{i=1}^{G}$ 独立计算组内归一化优势 $\hat{A}_i$
- 仅更新对应模块 $M$ 的 LM 权重 $\theta_M$
- 使用 [[LoRA]] 进行参数高效微调（rank=16），仅更新 q, k, v, o, up, down, gate 投影矩阵

#### 模块3: BetterTogether（提示优化 + 策略梯度组合）

**设计动机**: [[提示优化]]（PO）和 RL 权重优化提供互补收益——PO 优化输入分布，RL 优化模型行为。

**具体实现**:
- **Stage 1 — Prompt Optimization**: 使用 MIPROv2 优化各模块的提示模板 $\pi_M$
- **Stage 2 — Weight Optimization**: 使用 mmGRPO 对 LM 权重 $\theta_M$ 进行在线策略梯度微调
- 支持教师程序：可引入固定教师轨迹进行离线训练或 warm-start

---

## 关键公式

### 公式1: [[GRPO]] 标准目标函数

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\{(q, o_i, r_i)\}_{i=1}^{G}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( \omega_t \hat{A}_i, \operatorname{clip}(\omega_t, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}[p_\theta \| p_{\theta_{\text{ref}}}] \right\} \right]
$$

**含义**: 标准 GRPO 目标函数，对组内每个 response 的每个 token 计算 clipped importance ratio 乘以优势，并加入 [[KL散度]] 正则项。

**符号说明**:
- $G$: 组大小（每条输入采样 response 数）
- $|o_i|$: 第 $i$ 条 response 的 token 长度
- $\omega_t = \frac{p_\theta(o_{i,t} \mid q, o_{i,<t})}{p_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$: token 级 [[Importance Sampling|重要性采样]] 比率
- $\epsilon$: 裁剪超参数（限制策略更新幅度）
- $\hat{A}_i$: 组内归一化优势
- $\beta$: [[KL散度]] 惩罚系数

### 公式2: [[GRPO]] 组内优势归一化

$$
\hat{A}_i = \frac{r_i - \text{mean}(\mathcal{R})}{\text{std}(\mathcal{R})}, \quad \mathcal{R} = \{r_i\}_{i=1}^{G}
$$

**含义**: 将组内所有 response 的奖励归一化为零均值单位方差的相对优势，消除奖励尺度影响。

**符号说明**:
- $r_i$: 第 $i$ 条轨迹的标量奖励
- $\text{mean}(\mathcal{R})$: 组内奖励均值
- $\text{std}(\mathcal{R})$: 组内奖励标准差

### 公式3: mmGRPO 模块级目标函数

$$
\mathcal{J}_{\text{mmGRPO}}(\theta_M) = \mathbb{E}_{\{(q_i, o_i, r_i)\}_{i=1}^{G}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( \omega_t \hat{A}_i, \operatorname{clip}(\omega_t, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}[p_{\theta_M} \| p_{\theta_{M_{\text{ref}}}}] \right\} \right]
$$

**含义**: mmGRPO 的模块级目标函数。与标准 GRPO 的关键区别在于：更新目标仅为模块 $M$ 的 LM 权重 $\theta_M$，且组内各样本的 prompt $q_i$ 可能不同（来自不同轨迹）。

**符号说明**:
- $\theta_M$: 模块 $M$ 的 LM 权重（仅更新该模块的 LoRA 参数）
- $q_i$: 模块 $M$ 在第 $i$ 条轨迹中的输入 prompt（跨轨迹可能不同）
- $o_i$: 模块 $M$ 在第 $i$ 条轨迹中的输出
- $\omega_t = \frac{p_{\theta_M}(o_{i,t} \mid q_i, o_{i,<t})}{p_{\theta_{M_{\text{old}}}}(o_{i,t} \mid q_i, o_{i,<t})}$: 模块 $M$ 的 importance ratio

### 公式4: mmGRPO 策略梯度形式

$$
\nabla_{\theta_M} \mathcal{J}_{\text{mmGRPO}}(\theta_M) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \hat{A}_i + \beta \left( \frac{p_{\theta_{M_{\text{ref}}}}(o_{i,t}|q_i,o_{i,<t})}{p_{\theta_M}(o_{i,t}|q_i,o_{i,<t})} - 1 \right) \right) \nabla_{\theta_M} \log p_{\theta_M}(o_{i,t}|q_i,o_{i,<t}) \right]
$$

**含义**: mmGRPO 的梯度形式——一个重加权策略梯度：高优势轨迹 token 被上加权，低优势轨迹 token 被下加权（通过 clipping 机制），并通过 KL 惩罚向参考策略正则化。

**符号说明**:
- $\nabla_{\theta_M} \mathcal{J}_{\text{mmGRPO}}$: 目标函数关于模块 $M$ 权重的梯度
- $\log p_{\theta_M}(o_{i,t}|q_i,o_{i,<t})$: 模块 $M$ 生成第 $i$ 条轨迹第 $t$ 个 token 的对数概率
- $\beta(\frac{p_{\theta_{M_{\text{ref}}}}}{p_{\theta_M}} - 1)$: KL 惩罚的梯度贡献

### 公式5: [[Importance Sampling|重要性采样]] 局部马尔可夫分解

$$
p(o_{i,t} \mid \rho_{<i}, q_i, o_{i,<t}) = p_M(o_{i,t} \mid q_i, o_{i,<t})
$$

**含义**: 在确定性程序路由假设下（所有随机性仅来自 LM 生成），模块 $M$ 的 token 条件概率仅依赖于模块的局部上下文，与历史轨迹无关。这使得 importance ratio 可以仅在模块级别计算。

**符号说明**:
- $\rho_{<i}$: 第 $i$ 条轨迹的历史上下文
- $p_M$: 模块 $M$ 的策略分布

---

## 关键图表

### Figure 1: mmGRPO 系统概览

![Figure 1: mmGRPO Overview](https://arxiv.org/html/2508.04660v2/x1.png)

**说明**: mmGRPO 的整体架构。从多模块 LM 程序中采样多条轨迹，按模块标识符和调用顺序将模块调用分组，对每组独立应用 GRPO 策略更新。BetterTogether 框架在此基础上加入提示优化（MIPROv2）作为第一阶段。

### Algorithm 1: mmGRPO — 多模块 LM 程序的 GRPO

**伪代码思路**:
```
1. 输入: LM程序 Φ, 模块集 M, 输入x, 组大小G
2. 对每条样本:
   a. 采样 G 条完整轨迹 (y, ρ)
   b. 计算程序级奖励 r = μ(y, ρ, m)
   c. 调用 FormModuleLevelGroups 将模块调用分组
   d. 对每个模块组独立计算 GRPO 损失
   e. 更新对应模块的 LoRA 权重
3. 返回: 优化后的多模块 LM 程序
```

### Algorithm 2: FormModuleLevelGroups — 创建模块级 GRPO 分组

**伪代码思路**:
```
1. 初始化: groups = {} (以 (module_id, relative_index) 为键)
2. 遍历: 所有轨迹的模块调用序列
3. 分组: 按 (module_id, relative_index) 将 (q, o, r) 加入对应组
4. PAD_GROUPS: 使用 fill/truncate 策略处理变长轨迹
5. SELECTK_DIVERSE_ELEMENTS: 选择最大化奖励方差的 K 个元素
6. 返回: 对齐后的模块级 GRPO 组
```

### Table 1: 主要结果

| Strategy | Banking77 | PAPILLON | HoVer4-HOP | Avg Score |
|----------|:---------:|:--------:|:-----------:|:---------:|
| Vanilla CoT | 61.5% | 77.3% | 60.1% | 66.3% |
| MIPROv2 (PO) | 62.7% | 81.0% | 66.4% | 70.0% |
| mmGRPO | 64.3% | 83.6% | 65.6% | 71.2% |
| **BetterTogether (PO + mmGRPO)** | **66.4%** | **83.8%** | **69.9%** | **73.4%** |

*注: 数值为 llama3.1-8b 和 qwen3-8b 在 3 个种子上的平均值*

**关键发现**:
- BetterTogether 相比 Vanilla CoT 提升约 **11%**
- 相比纯提示优化（MIPROv2）提升约 **5%**
- 相比纯 mmGRPO 提升约 **3%**

### Table 2: 训练超参数

| 超参数 | 值 |
|--------|------|
| Temperature | 0.6 |
| Learning rate | $1 \times 10^{-5}$ |
| Gradient accumulation steps | 20 |
| Per-device train batch size | 1 |
| KL penalty $\beta$ | 0.01 (qwen3-8b) / 0.04 (llama3.1-8b) |
| Gradient norm clipping | 0.1 (qwen3-8b) / 0.5 (llama3.1-8b) |
| Training steps | 750 |
| Examples per step | 4 |
| Rollouts per example | 12 |
| GRPO group size $G$ | 12 |
| [[LoRA]] rank $r$ | 16 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| LoRA targets | q, k, v, o, up, down, gate |
| Max context length | 8,192 tokens |
| GPU hours (mmGRPO) | 18.7 hrs (2x H100) |
| GPU hours (MIPROv2) | 1.4 hrs (1x H100) |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Banking77 | ~13k | 77 类银行意图分类 | 分类任务评估 |
| [[PAPILLON]] | ~5k | 隐私保护委派（PII 脱敏 + 信息检索） | 多步骤隐私敏感任务 |
| HoVer4-HOP | ~18k | 多跳事实验证（4跳） | 多步检索推理 |

### 实现细节

- **Backbone**: Llama 3.1-8B-Instruct / Qwen3-8B
- **优化器**: AdamW，lr=$1 \times 10^{-5}$
- **Batch Size**: 4 examples/step，12 rollouts/example
- **训练轮数**: 750 steps
- **硬件**: 2x H100 (mmGRPO)，1x H100 (MIPROv2)
- **微调方式**: [[LoRA]] rank=16, alpha=64, dropout=0.05
- **框架**: DSPy (`dspy.GRPO` optimizer) + Arbor

### 关键发现

1. **模块级 vs 轨迹级分组**: 模块级分组在变长轨迹场景下优于简单轨迹级分组，因为可以更细粒度地对齐结构相同的模块调用
2. **多样性选择收益**: `SelectKDiverseElements` 通过最大化组内奖励方差提升梯度的判别能力，尤其在早期训练阶段
3. **PO 顺序重要性**: 先 PO 后 mmGRPO 优于相反顺序，提示优化为 RL 提供了更好的初始分布
4. **计算效率**: PO（1.4 GPU-hours）远低于 mmGRPO（18.7 GPU-hours），但两者组合带来最佳性能

---

## 批判性思考

### 优点
1. **问题选择精准**: 多模块 LM 程序是实际应用中的常见形态，扩展 GRPO 到该场景有广泛实用性
2. **模块级分组的理论优雅性**: 在确定性控制流下，模块级和轨迹级梯度方向一致，但模块级分组更具灵活性（能处理变长轨迹）
3. **BetterTogether 框架的互补性**: PO 优化输入分布 + RL 优化模型行为的组合天然互补，实验验证了累加收益
4. **全面开源**: 作为 `dspy.GRPO` 集成到 DSPy 库，代码、教程、文档齐全

### 局限性
1. **统一信用分配过于简单**: 所有模块共享最终程序奖励，无法区分各模块对最终结果的独立贡献，可能导致次优的模块级更新
2. **确定性子路由假设限制**: 论文假设控制流是确定性的（随机性仅来自 LM），实际多模块程序可能包含非确定性路由（如条件分支的随机选择）
3. **计算开销**: mmGRPO 需要 18.7 GPU-hours，远高于 PO（1.4 小时），在大规模部署时成本较高
4. **单共享 LM 限制**: 当前实现仅支持所有模块共享同一 LM，无法处理不同模块使用不同 LM 的场景

### 潜在改进方向
1. **模块级信用分配**: 引入中间奖励或差分信用分配（如 [[TACO]] 的 token 级校准扩展到模块级）
2. **异步模块更新**: 借鉴 [[SAO]] 的思路，探索不需要等完整组的异步模块级更新
3. **多 LM 支持**: 扩展到不同模块使用不同 LM 的场景，混合不同规模的模型
4. **非确定性控制流**: 将控制流决策也纳入策略梯度的优化范围

### 可复现性评估
- [x] 代码开源（DSPy 库 `dspy.GRPO`）
- [x] 预训练模型（Llama 3.1-8B, Qwen3-8B）
- [x] 训练细节完整（超参数表详细列出）
- [x] 数据集可获取（Banking77, PAPILLON, HoVer 均为公开数据集）

---

## 关联笔记

### 基于
- [[GRPO]]: mmGRPO 的基础算法，将 GRPO 从单模块扩展到多模块
- [[PPO]]: GRPO 和 mmGRPO 的算法根源（clipped surrogate objective）
- [[BetterTogether]]: 提示优化与策略梯度组合的前身工作（Soylu et al., 2024）

### 对比
- [[SAO]]: 替代 GRPO 的异步单 rollout 方法，解决 GRPO 的同步等待问题
- [[TACO]]: 在 GRPO 基础上增加 token-level credit 校准
- MIPROv2: 提示优化基线，与 mmGRPO 组合成 BetterTogether

### 方法相关
- [[Importance Sampling]]: mmGRPO 使用 token 级 IS ratio 计算策略梯度
- [[KL散度]]: mmGRPO 目标函数中用于约束策略更新的正则项
- [[LoRA]]: mmGRPO 实现参数高效微调的方式
- [[DSPy]]: mmGRPO 的开源实现框架

### 任务相关
- [[PAPILLON]]: 隐私保护委派任务，mmGRPO 评估任务之一
- [[CoT]]: 基线方法使用的推理策略

---

## 速查卡片

> [!summary] Multi-module GRPO: Composing Policy Gradients and Prompt Optimization for LM Programs
> - **核心**: 将 GRPO 扩展到多模块 LM 程序，通过模块级分组策略梯度实现变长/异构轨迹的策略优化
> - **方法**: mmGRPO（模块级 GRPO）+ BetterTogether（先 MIPROv2 提示优化，再 mmGRPO 权重微调）
> - **结果**: BetterTogether 平均 73.4% 准确率，较 Vanilla CoT 提升 11%，较 PO alone 提升 5%，较 mmGRPO alone 提升 3%
> - **代码**: https://github.com/stanfordnlp/dspy (dspy.GRPO)

---

*笔记创建时间: 2026-07-15 15:00*
