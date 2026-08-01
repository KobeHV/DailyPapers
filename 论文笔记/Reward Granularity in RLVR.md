---
title: "Reward Granularity in RLVR: Comparing Process and Outcome Reward Structures for Mathematical Reasoning in Small Language Models"
method_name: "Reward Granularity in RLVR"
authors: [Anagha Radhakrishna Palandye, Rebecca Glick, Osheen Kaul]
year: 2026
venue: arXiv
tags: [rlvr, process-reward, mathematical-reasoning, grpo, outcome-reward, reward-granularity, small-language-models]
zotero_collection: 
image_source: local
arxiv_html: https://arxiv.org/html/2607.02869
created: 2026-08-01
---

# 论文笔记：Reward Granularity in RLVR

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | New York University |
| 日期 | July 2026 |
| 项目主页 | — |
| 对比基线 | Base (no training, no CoT) |
| 链接 | [arXiv](https://arxiv.org/abs/2607.02869) |

---

## 一句话总结

> 系统比较过程级与结果级奖励对小型语言模型数学推理的影响，证明 [[Reward Granularity|奖励粒度]] 是 [[RLVR]] 中的一等设计决策，过程监督比结果监督准确率高出近 10 个百分点。

---

## 核心贡献

1. **系统比较五种奖励条件**: 在 [[Qwen2.5]]-0.5B + [[GRPO]] + [[GSM8K]] 设定下，对比纯过程、纯结果、三种混合权重的效果。
2. **超越最终答案准确率的评估**: 引入 [[CoT-Pass@K|CoT-Pass@1]]、Trace Validity、Chain Length Deviation、Process Step Ratio 四个推理质量指标。
3. **LLM-as-a-Judge 错误分析**: 用 [[GPT-4o]] 标注不同奖励机制下的错误模式分布，揭示过程模型与结果模型的截然不同的失败模式。
4. **发现混合奖励异常**: 低过程权重 ($\lambda=0.1$) 的混合配置反而劣于纯结果监督，说明存在冲突的优化信号。

---

## 问题背景

### 要解决的问题

小语言模型在使用 [[Chain-of-Thought]] 推理时，经常生成不正确、不一致或逻辑混乱的中间步骤。核心问题是：**数学推理能力的提升是真正的推理质量提升，还是模型仅仅是更好地采样了看似合理的答案？**

### 现有方法的局限

大多数 [[RLVR]] 研究只奖励最终答案（[[Outcome Reward|结果级奖励]]），忽略了步骤级过程监督（[[Process Reward|过程级奖励]]）的影响。小模型尤其缺乏在稀疏反馈下自我纠正的能力。

现有工作（Wen et al. 2025; Tang et al. 2025）虽在 RLVR 中取得了进展，但几乎全部只使用最终答案奖励，**奖励粒度**（reward granularity）这一维度尚未被系统研究。

### 本文的动机

过程奖励和结果奖励各有利弊：过程奖励提供密集监督但可能导致过于冗长的推理链，结果奖励高效但让推理过程不透明。两者混合是否能结合各自优势？这一问题对缺乏内部纠错能力的小模型尤为重要。

---

## 方法详解

### 模型架构

本文不是提出新架构，而是系统比较不同 [[Reward Granularity|奖励粒度]] 对推理质量的影响。整体采用 **[[RLVR]] + [[GRPO]]** 训练框架：

- **Base Model**: [[Qwen2.5]]-0.5B (494M params)
- **训练框架**: [[VERL]] + [[vLLM]]
- **RL 算法**: [[GRPO]]（[[PPO]] 的变体，专为语言模型微调设计）
- **数据集**: [[GSM8K]]（7,473 训练 / 1,319 测试）
- **奖励设计**: 五种条件（见核心模块）

### 核心模块

#### 模块1: [[Process Reward|过程奖励]] (Process Supervision)

**设计动机**: 在每个中间步骤提供密集监督信号，鼓励模型产生逐步可验证的正确推理。

**具体实现**:
- 从 [[GSM8K]] 解答中提取参考答案步骤
- 模型生成的每一步与参考答案步骤比较（数值容差 $10^{-5}$）
- 每步匹配得 1，否则得 0
- 总过程奖励 = 正确步数 / 总步数
- 当生成链长度超过参考答案 1.5 倍时施加惩罚，防止过度冗长

#### 模块2: [[Outcome Reward|结果奖励]] (Outcome Supervision)

**设计动机**: 仅关注最终答案正确性，让模型自主探索推理路径。

**具体实现**:
- 从模型响应中提取最终答案（支持 `\boxed{}`、`#### answer`、自然语言格式）
- 与参考答案进行数值比较（容差 $10^{-5}$）
- 二元奖励：匹配得 1，否则得 0

#### 模块3: [[Reward Granularity|混合奖励]] (Hybrid Rewards)

**设计动机**: 结合过程监督的推理质量与结果监督的答案导向，获得更稳定的优化信号。

**具体实现**:
- 加权组合：$R = \lambda \cdot R_{\text{process}} + (1-\lambda) \cdot R_{\text{outcome}}$
- 三种混合配置：$\lambda \in \{0.9, 0.5, 0.1\}$
- $\lambda=0.9$：重过程 + 轻结果引导
- $\lambda=0.5$：均衡监督
- $\lambda=0.1$：重结果 + 最小过程监督

---

## 关键公式

### 公式1: [[Process Reward|过程奖励函数]]

$$
R_{\text{process}} = \frac{\text{correct steps}}{\text{total steps}}
$$

**含义**: 计算推理链中与参考答案匹配的步骤比例，提供步骤级密集反馈。

**符号说明**:
- $\text{correct steps}$: 与参考答案步骤匹配（数值容差 $10^{-5}$）的步数
- $\text{total steps}$: 模型生成的总步数
- 额外惩罚：当生成链长度 $> 1.5 \times$ 参考答案步数时触发

### 公式2: [[Outcome Reward|结果奖励函数]]

$$
R_{\text{outcome}} = \begin{cases} 1 & \text{if } |\hat{a} - a| < \epsilon = 10^{-5} \\ 0 & \text{otherwise} \end{cases}
$$

**含义**: 仅根据最终数值答案是否正确给予二元奖励。

**符号说明**:
- $\hat{a}$: 模型预测的最终答案
- $a$: 参考答案
- $\epsilon$: 数值容差，设为 $10^{-5}$

### 公式3: [[Reward Granularity|混合奖励函数]]

$$
R = \lambda \cdot R_{\text{process}} + (1 - \lambda) \cdot R_{\text{outcome}}
$$

**含义**: 通过加权组合融合过程和结果两种监督信号，$\lambda$ 控制过程奖励的相对权重。

**符号说明**:
- $\lambda$: 过程奖励权重，取 $\{0.9, 0.5, 0.1\}$
- $R_{\text{process}}$: 过程奖励，来自公式 1
- $R_{\text{outcome}}$: 结果奖励，来自公式 2
- $1-\lambda$: 结果奖励权重

---

## 关键图表

### Figure 1: RLVR Training Pipeline / 训练流程概览

![[RewardGranularity_RLVR_Pipeline.png]]

**说明**: [[RLVR]] 训练管线的端到端视图。Base Model [[Qwen2.5]]-0.5B 通过 [[vLLM]] 生成 rollouts，由奖励函数（过程/结果/混合）评估，再通过 [[GRPO]] 在 [[VERL]] 框架内更新策略。

### Figure 2: Reasoning Traces Comparison / 推理链对比

![[RewardGranularity_LLM_Reasoning_Traces.png]]

**说明**: 四个奖励机制在同一个几何题上的推理链对比。Base 和 Outcome 错误计算了面积（300 sq ft），而 Process 和 Process+Outcome 正确识别需要计算周长（70 ft）。过程奖励通过强制显式变量定义（长度/宽度）来减少推理错误。

### Figure 3: Error Type Distribution / 错误类型分布

![[RewardGranularity_fig3_error_dist.png]]

**说明**: [[GPT-4o]] 在 45 题评估切片上标注的错误类型分布。过程模型频繁引入结构不一致和矛盾，结果模型表现出推导和算术错误，混合模型错误分布最均匀。

### Figure 4: Training Curves / 训练曲线

![[RewardGranularity_fig4_training_curves.png]]

**说明**: 五种奖励条件在 [[GSM8K]] 上的训练曲线。Process-only 收敛到最高验证准确率（63.73%），而 Process (0.1) + Outcome (0.9) 始终低于所有其他条件包括纯结果监督，展示了混合奖励异常现象。

### Table 1: Test Set Distribution / 测试集步数分布

| 参考答案步数 | GSM8K 题目数量 |
|-------------|---------------|
| 2 | 326 |
| 3 | 370 |
| 4 | 298 |
| 5 | 174 |
| 6 | 88 |
| 7 | 40 |
| 8 | 20 |
| 9 | 2 |
| 11 | 1 |
| **Total** | **1,319** |

**说明**: 测试集按参考答案步数分层。基于此分布构建了 45 题平衡评估切片（简单 15 + 中等 15 + 困难 15）。

### Table 2: Test Set Accuracy / 测试集准确率

| Regime | Test Acc. (%) | Improvement over Baseline (%) |
|--------|---------------|-------------------------------|
| Base (no training) | 33.13 | — |
| Process Only | **63.73** | **+30.60** |
| Outcome Only | 53.75 | +20.62 |
| Process (0.5) + Outcome (0.5) | 57.40 | +24.27 |
| Process (0.9) + Outcome (0.1) | 61.10 | +27.97 |
| Process (0.1) + Outcome (0.9) | 49.30 | +16.17 |

**说明**: Process-only 达到最高准确率 63.73%。Process (0.9) + Outcome (0.1) 是最佳混合配置（61.10%）。Process (0.1) + Outcome (0.9) 表现异常——虽为混合方案，却低于纯 Outcome（49.30% vs 53.75%），显示冲突优化信号。

### Table 3: Reasoning Fidelity Metrics / 推理忠实度指标

| Regime | CoT-Pass@1* | Trace Validity | Chain Length Dev. | Process Step Ratio |
|--------|-------------|----------------|-------------------|-------------------|
| Base | 0.33 | 0.56 | 3.00 | 0.79 |
| Process Only | **0.64** | 0.22 | 6.49 | 0.64 |
| Outcome Only | 0.54 | 0.40 | 3.64 | 0.77 |
| Process (0.9) + Outcome (0.1) | 0.61 | **0.60** | **3.49** | **0.84** |

*CoT-Pass@1 在完整 1,319 题测试集上报告；其余在 45 题分层评估切片上计算。加粗 = RLVR 训练条件中的最佳结果。

**说明**: Process-only 在 CoT-Pass@1 上最高（0.64），但 Trace Validity 最低（0.22）且 Chain Length Deviation 最大（6.49）——反映其冗长推理链的代价。Process (0.9) + Outcome (0.1) 在 Trace Validity、Chain Length Dev.、Process Step Ratio 三项上全面最优，展示了混合奖励的平衡优势。

### Table 4: GRPO Training Hyperparameters / 训练超参数

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-0.5B (494M params) |
| Train samples | 7,473 |
| Test samples | 1,319 |
| Max prompt / response | 512 / 1,024 tokens |
| Batch size / mini-batch | 64 / 8 |
| Learning rate | $1 \times 10^{-6}$ |
| KL coefficient | 0.001 |
| Epochs / total steps | 5 / 580 |
| Rollouts / temperature (train) | 5 / 0.6 |
| GPU | 1x A100 80GB |
| Framework | VERL, vLLM |

**说明**: 所有奖励条件使用完全相同的超参数，确保实验的公平比较。所有实验使用单张 A100 80GB GPU。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[GSM8K]] | 8,792 题 | 小学数学文字题，需逐步算术推理 | 7,473 训练 / 1,319 测试 |
| 45-question evaluation slice | 45 题 | 简单15 (2-3步) + 中等15 (4-5步) + 困难15 (6-11步) | 推理质量深度分析 |

### 实现细节

- **Backbone**: [[Qwen2.5]]-0.5B (494M parameters)
- **RL 算法**: [[GRPO]]（通过 [[VERL]] 实现）
- **推理引擎**: [[vLLM]]（5 rollouts/prompt, temperature 0.6）
- **Learning Rate**: $1 \times 10^{-6}$
- **KL 系数**: 0.001
- **Batch Size**: 64 (mini-batch 8)
- **训练轮数**: 5 epochs (580 steps)
- **序列长度**: prompt 512 / response 1,024 tokens
- **硬件**: 单张 A100 80GB GPU

### 主要发现

1. **过程奖励大幅领先**: Process-only（63.73%）vs Outcome-only（53.75%），近 10 个百分点差距
2. **混合奖励的异常**: $\lambda=0.1$ 配置（49.30%）反而低于纯结果监督（53.75%），说明低比例过程信号可能引入干扰而非帮助
3. **推理质量 vs 准确率的权衡**: 纯过程奖励准确率最高但 Trace Validity 最低（0.22），混合 $\lambda=0.9$ 在准确率（61.10%）和推理忠实度（Trace Validity 0.60）之间取得最佳平衡
4. **错误模式分化**: 过程模型犯结构不一致/矛盾类错误（因过长推理链），结果模型犯推导/算术类错误（因缺乏中间步骤验证）

---

## 批判性思考

### 优点
1. **研究问题精准**: 奖励粒度在 RLVR 中是一个被忽视但至关重要的维度，本文选题切中要害
2. **实验设计系统**: 五种条件全覆盖（基线、纯过程、纯结果、三种混合），超参数严格控制，比较公平
3. **多维度评估**: 不只看最终答案准确率，还引入 CoT-Pass@1、Trace Validity、Chain Length Deviation、Process Step Ratio 四个推理质量指标
4. **混合奖励异常的发现**: $\lambda=0.1$ 低于纯结果的发现具有反直觉价值，揭示了过程/结果奖励并非简单的加法关系

### 局限性
1. **单次运行无统计显著性**: 所有结果来自单次训练运行，无多 seed 方差报告，近 10% 的差距应视为指示性而非统计确认的
2. **模型和数据集的局限**: 仅在 Qwen2.5-0.5B + GSM8K 上验证，泛化性未知。GSM8K 为小学数学题，结论是否适用于更复杂的数学（MATH）或跨领域推理未知
3. **45 题评估切片偏小**: 虽然平衡了难度分层，但样本量有限，可能不足以捕捉全部行为差异
4. **GPT-4o 法官的可靠性**: LLM-as-a-Judge 的标注可能无法完全反映人类评估结果
5. **步级验证在数学中易但在开放领域难**: 数学题的步骤验证相对明确，结论能否推广到需要主观判断的推理任务存疑

### 潜在改进方向
1. **扩展到更大模型和更难数据集**: 在 Qwen2.5-7B/14B + MATH/SVAMP/AQuA-RAT 上验证结论；探索多跳推理（HotpotQA）
2. **部分分奖励机制**: 设计更细粒度的奖励（如 partial-credit），替代二元匹配
3. **学习型验证器**: 训练专门的 PRM（Process Reward Model）替代规则匹配，可能提供更丰富的监督信号
4. **多 seed 统计验证**: 重新运行多次实验以提供置信区间
5. **跨领域的奖励粒度研究**: 扩展到代码生成、符号推理等任务

### 可复现性评估
- [ ] 代码开源（承诺发表后发布）
- [ ] 预训练模型（使用公开的 Qwen2.5-0.5B）
- [x] 训练细节完整（Table 4 包含所有超参数）
- [x] 数据集可获取（GSM8K 公开可用）
- [ ] 多 seed 运行（仅单次运行）

---

## 关联笔记

### 基于
- [[RLVR]]: 本文直接建立在可验证奖励的强化学习框架之上
- [[GRPO]]: 使用的核心 RL 算法
- [[Chain-of-Thought|CoT]]: 研究的推理范式
- [[Qwen2.5]]: 使用的基座模型

### 对比
- [[RLVR]] 相关工作（Wen et al. 2025, Tang et al. 2025）: 使用了结果奖励但未探索奖励粒度
- [[Process Reward|Process Reward 方法]]（Lightman et al. 2023, Uesato et al. 2022）: 早期过程监督探索
- Samineni et al. (2025): 在相同模型/数据集上研究 RLVR 局部一致性但未改变奖励粒度

### 方法相关
- [[Reward Granularity]]: 核心研究概念
- [[Process Reward]]: 过程级监督信号设计
- [[Outcome Reward]]: 结果级监督信号设计
- [[GRPO]]: Group Relative Policy Optimization
- [[VERL]]: 训练框架
- [[vLLM]]: 推理引擎

### 硬件/数据相关
- [[GSM8K]]: 小学数学文字题基准
- [[GPT-4o]]: 用于 LLM-as-a-Judge 错误分析

---

## 速查卡片

> [!summary] Reward Granularity in RLVR
> - **核心**: 系统证明奖励粒度是 RLVR 的一等设计决策，过程级监督在小型语言模型的数学推理上大幅优于结果级监督
> - **方法**: Qwen2.5-0.5B + GRPO on GSM8K，五种奖励条件（纯过程/纯结果/三种混合）
> - **结果**: Process-only 63.73% vs Outcome-only 53.75%；$\lambda=0.9$ 混合在准确率和推理忠实度间取得最佳平衡；$\lambda=0.1$ 出现异常反转
> - **代码**: 承诺发表后发布

---

*笔记创建时间: 2026-08-01*
