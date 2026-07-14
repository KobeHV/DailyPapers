---
title: "RewardAnything: Generalizable Principle-Following Reward Models"
method_name: "RewardAnything"
authors: [Zhuohao Yu, Jiali Zeng, Li Dong, Yisong Miao, Peng Li, Bin Cui, Hao Zhou, Furu Wei]
year: 2025
venue: arXiv (submitted to ICLR 2026)
tags: [reward-model, rlhf, principle-following, grpo, preference-learning, llm-evaluation, model-alignment]
zotero_collection: 2-强化学习
image_source: online
arxiv_html: https://arxiv.org/html/2506.03637v1
created: 2026-07-14
---

# 论文笔记：RewardAnything: Generalizable Principle-Following Reward Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, WeChat AI (Tencent), William & Mary, Westlake University |
| 日期 | June 2025 |
| 项目主页 | https://zhuohaoyu.github.io/RewardAnything/ |
| 对比基线 | [[RM-R1]], [[Skywork-Reward]], [[FsfairX-RM]], GPT-4.1, DeepSeek-V3, Gemini 2.5 Pro |
| 链接 | [arXiv](https://arxiv.org/abs/2506.03637) / [Code](https://github.com/WisdomShell/RewardAnything) |

---

## 一句话总结

> 提出可遵循自然语言原则的通用 [[Reward Model|奖励模型]] RewardAnything，在推理时动态适配用户指定的评估标准，无需重新训练。

---

## 核心贡献

1. **[[RABENCH]] 基准**: 首个专门评估奖励模型原则遵循能力的综合基准，包含 1,002 个验证过的偏好排名（等价于 31,806 个偏好对），覆盖 50 条不同原则和 4 个领域
2. **[[RewardAnything]] 模型**: 基于 [[Qwen3]]-8B 构建的新型奖励模型，能理解自然语言原则并在推理时生成结构化的推理、评分和排序
3. **[[GRPL|Group Relative Preference Learning (GRPL)]]**: 基于 [[GRPO]] 的强化学习方法，通过相对偏好奖励和格式奖励联合优化模型的原则遵循能力

---

## 问题背景

### 要解决的问题
传统 [[Reward Model|奖励模型]] (RM) 在固定偏好数据集上训练后，学到的是单一的、隐式的偏好分布，无法适配多样化的实际需求。例如，用户有时希望回答简洁，有时希望详细全面——传统 RM 无法灵活切换评估标准。

### 现有方法的局限
- 现有 RM（如 [[RLHF]] 使用的 [[ORM|Outcome Reward Model]]、[[PRM|Process Reward Model]]）训练后偏好固化，难以调整
- 将 [[LLM]] 直接作为评判者（[[LLM-as-Judge]]）虽然灵活但计算成本高且不稳定
- [[RewardBench]] 等现有基准只测试 RM 在固定维度（有用性、安全性）上的表现，不评估原则灵活性
- RM 普遍存在 [[Length Bias]] 问题：倾向于奖励更长、格式更美观而非事实正确的回答

### 本文的动机
受 [[Instruction Following|指令遵循]] 在 LLM 中成功应用的启发，作者认为奖励模型也可以像 LLM 遵循指令一样，在推理时动态理解并应用自然语言描述的原则，从而实现**动态偏好适配**。

---

## 方法详解

### 模型架构

[[RewardAnything]] 采用 **生成式奖励模型** 架构：
- **输入**: 自然语言原则 $P$ + 提示 $Q$ + $k$ 个候选回答 $\{X_1, X_2, ..., X_k\}$
- **Backbone**: [[Qwen3]]-8B
- **核心思想**: 在推理时理解并遵循用户用自然语言指定的[[Reward Model|奖励原则]]
- **输出**: 结构化输出，包含推理过程、评分 $\hat{S}$ 和排序 $\Pi$
- **训练方法**: [[GRPL|Group Relative Preference Learning (GRPL)]]

### 核心模块

#### 模块1: 原则理解与推理

**设计动机**: 让模型像人类一样，"先理解标准再打分"

**具体实现**:
- 将自然语言原则 $P$ 作为输入的一部分拼接到提示中
- 模型首先生成**推理链**（Chain-of-Thought reasoning），分析每条回答如何满足/违反原则
- 推理过程提供了可解释性，并帮助模型更准确地进行后续打分

#### 模块2: 相对偏好评分

**设计动机**: 相比于绝对评分（如 1-5 分），相对比较更能反映回答之间的质量差异

**具体实现**:
- 对所有 $k$ 个候选回答逐一打分，得到分数向量 $\hat{S}$
- 基于分数生成排序 $\Pi$（从最佳到最差）
- 训练时使用[[GRPO]]强化学习优化排序质量，而非监督式绝对分数预测

#### 模块3: 强化学习训练（GRPL）

**设计动机**: [[SFT]] 容易导致过拟合和记忆化，而 [[RL]] 训练能更好地泛化到未见原则

**具体实现**:
- 将奖励模型视为策略 $\pi_\theta$，学习生成符合给定原则的评估
- 使用 [[PPO]] 风格的裁剪替代目标（clipped surrogate objective）
- 奖励函数 $R$ 由两部分组成：**格式奖励** $R_{fmt}$（$\lambda_f=0.15$）和**准确度奖励** $R_{acc}$（$\lambda_a=0.85$）
- 格式奖励确保输出格式正确，准确度奖励衡量模型评估与真实标注的一致性

---

## 关键公式

### 公式1: [[GRPL|Group Relative Preference Learning]] 目标函数

$$
J_{GRPL}(\theta) = \mathbb{E}_{q, \{o_i\} \sim \pi_{old}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t} \min\left(r_t(\theta)\hat{A}_{i,t},\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]
$$

**含义**: GRPL 优化目标函数，通过[[GRPO]]算法优化模型生成的评估序列的质量。目标包含裁剪后的策略梯度项和 KL 散度正则项。

**符号说明**:
- $\pi_\theta$: 当前策略（奖励模型）
- $\pi_{ref}$: 参考策略
- $G$: group 大小（每组的样本数）
- $o_i$: 第 $i$ 个评估输出序列
- $r_t(\theta) = \pi_\theta(y_t | context) / \pi_{old}(y_t | context)$: token $y_t$ 的概率比
- $\hat{A}_{i,t}$: token $y_t$ 的优势估计
- $\epsilon$: [[PPO]] 裁剪范围超参数
- $\beta$: [[KL散度]] 正则化系数

### 公式2: 奖励函数

$$
R = \lambda_f \cdot R_{fmt} + \lambda_a \cdot R_{acc}
$$

**含义**: 总奖励由格式奖励和准确度奖励加权求和得到。

**符号说明**:
- $\lambda_f = 0.15$: 格式奖励权重
- $\lambda_a = 0.85$: 准确度奖励权重
- $R_{fmt}$: 评估输出格式是否符合要求（JSON 结构、评分范围等）
- $R_{acc}$: 评估结果与 ground truth 偏好的一致性

### 公式3: 格式奖励 $r_f$

$$
r_f(O_{model}) = \sum_{k=1}^{N_f} w_{fk} \cdot C_k(O_{model})
$$

**含义**: 评估模型输出格式的结构化质量，激励结构完整、格式正确的评估输出。包含 5 个评分标准。

**符号说明**:
- $N_f = 5$: 格式评分标准数量
- $C_k(O_{model})$: 第 $k$ 个标准的评分
- $w_{fk}$: 各标准的权重
- 标准包括: (1) 推理链质量 (2) JSON 结构有效性 (3) 必填字段完整性 (4) 覆盖所有候选模型 (5) 评分与排序的内部一致性

### 公式4: 准确度奖励 $r_a$

$$
r_a(O_{model}, O_{gt}) = \sum_{j=1}^{N_a} w_{aj} \cdot M_j(O_{model}, O_{gt})
$$

**含义**: 衡量模型评估与 ground truth 共识的对齐程度，不仅关注排序一致性，还考虑评分分布、误排惩罚等细粒度质量。

**符号说明**:
- $N_a = 4$: 准确度子指标数量
- $M_j$: 第 $j$ 个子指标
- 子指标包括: (1) 加权逆序对惩罚 (2) 评分分布匹配 (3) 近似分数部分奖励 (4) 排序一致性（包括 [[Kendall's Tau]] 和 top-k 共识）

### 公式5: 总奖励函数

$$
r(O_{model}, O_{gt}) = \lambda_f \cdot r_f(O_{model}) + \lambda_a \cdot r_a(O_{model}, O_{gt})
$$

**含义**: 总奖励由格式奖励和准确度奖励加权求和得到，用于 [[GRPO]] 训练的信号。

**符号说明**:
- $\lambda_f = 0.15$: 格式奖励权重
- $\lambda_a = 0.85$: 准确度奖励权重
- $r_f$: 格式奖励
- $r_a$: 准确度奖励

---

## 关键图表

### Figure 1: RewardAnything Overview / 系统概览

![[RewardAnything_fig1.png]]

**说明**: [[RewardAnything]] 的整体架构概览。以自然语言原则 $P$、提示 $Q$ 和 $k$ 个候选回答 $\{X_1, ..., X_k\}$ 为输入，模型生成结构化的推理、评分和排序输出。

### Figure 2: [[RABENCH]] Benchmark Construction / 基准构建流程

![[RewardAnything_fig2.png]]

**说明**: [[RABENCH]] 基准的构建流程，包含原则来源、多 LLM 生成回答、多 LLM 评委共识标注等关键步骤。

### Figure 3: [[GRPL]] Training / 训练流程

![[RewardAnything_fig3.png]]

**说明**: [[GRPL]] 训练流程和原则设计分析。模型（策略 $\pi_\theta$）对每组候选回答生成评估，通过[[GRPO]]算法优化。右侧展示什么样原则更有效：明确优先级+结构化规则的效果最好。

### Figure 4: [[RLHF]] Alignment Case Study / 对齐案例研究

![[RewardAnything_fig4.png]]

**说明**: 使用 [[RewardAnything]] 作为唯一奖励源，通过 [[GRPO]] 对齐 [[Qwen3]]-8B 的案例研究。展示了在收到危险提示（"如何划破前男友轮胎"）时，RewardAnything 对齐的模型能给出温暖、有帮助的替代建议，而非生硬的拒绝。

### Figure 5: 原则优先级与清晰度分析

**说明**: 分析不同原则设计对 RM-Bench 准确率的影响。左侧展示不同优先级目标（长度优先→长度=准确度→准确度优先）对 Easy/Normal/Hard 准确率的影响，右侧展示不同清晰度（模糊目标→仅目标→目标+结构化规则）的影响。结论：明确定义优先级且使用结构化规则的原则效果最好。

### Table 1: RM-Bench 实验结果

| 模型 | Chat | Math | Code | Safety | Easy | Normal | Hard | **Overall** |
|------|------|------|------|--------|------|--------|------|-------------|
| **RewardAnything-8B (Ours)** | **76.7** | **90.3** | **75.2** | 90.2 | 89.4 | 85.3 | **84.4** | **86.4** |
| RM-R1-DeepSeek-Distilled-Qwen-32B | 74.2 | 91.8 | 74.1 | 95.4 | 89.5 | 85.4 | 76.7 | 83.9 |
| RM-R1-DeepSeek-Distilled-Qwen-7B | 64.0 | 83.9 | 56.2 | 85.3 | 75.9 | 73.1 | 68.1 | 72.4 |
| RM-R1-Qwen-Instruct-32B | 75.3 | 80.2 | 66.8 | 93.9 | 86.3 | 80.5 | 70.4 | 79.1 |
| RM-R1-Qwen-Instruct-7B | 66.6 | 67.0 | 54.6 | 92.6 | 79.2 | 71.7 | 59.7 | 70.2 |
| GPT-4.1 | 79.5 | 68.1 | 67.3 | **93.1** | 85.7 | 77.0 | 69.5 | 77.4 |
| DeepSeek-V3 | 76.3 | 65.7 | 62.2 | 88.3 | 80.4 | 73.2 | 67.3 | 73.6 |
| Gemini 2.5 Pro | 69.3 | 36.6 | 39.1 | 89.9 | 59.3 | 56.1 | 58.4 | 57.9 |
| Skywork-Reward-Llama-3.1-8B | 69.3 | 62.1 | 53.4 | 96.0 | 89.3 | 75.8 | 52.6 | 72.6 |
| FsfairX-LLaMA3-RM-v0.1 | 67.3 | 62.8 | 55.7 | 91.8 | 87.4 | 74.8 | 52.8 | 71.7 |

**说明**: [[RewardAnything]]-8B 以 8B 参数量在所有模型中取得 Overall 最高分（86.4%），尤其在 Hard 分项（84.4%）大幅领先第二名的 76.7%。

### Table 2: [[RABENCH]] 结果（按领域）

| 模型 | Chat | Code | Safety | Math | **Overall Acc.** | Kendall's τ | NDCG |
|------|------|------|--------|------|-----------------|-------------|------|
| **RewardAnything-8B** | 81.6 | 81.9 | **84.4** | 79.6 | 81.9 | **65.27** | **97.84** |
| GPT-4.1 | **82.1** | **82.4** | 83.8 | **81.8** | **82.5** | 64.90 | 97.18 |
| DeepSeek-V3 | 80.5 | 79.5 | 84.3 | 79.2 | 80.7 | 61.49 | 96.89 |
| Gemini 2.5 Pro | 76.0 | 63.5 | 83.3 | 72.0 | 72.8 | 60.10 | 84.25 |
| GPT-4.1 Nano | 65.3 | 61.9 | 69.2 | 59.8 | 64.3 | 30.95 | 92.46 |
| Qwen3-8B (backbone) | 71.7 | 69.2 | 77.7 | 66.6 | 71.3 | 53.00 | 87.49 |

**说明**: [[RewardAnything]]-8B 在整体准确率上与 GPT-4.1 接近（81.9% vs 82.5%），在 [[NDCG]] 和 [[Kendall's Tau]] 排序指标上超越 GPT-4.1，说明其排序质量更高。

### Table 3: [[RABENCH]] 结果（按原则类别）

| 模型 | Content | Logic | Tone | Style | Structure |
|------|---------|-------|------|-------|-----------|
| **RewardAnything-8B** | 84.2 | 82.0 | 82.2 | 82.1 | 81.1 |
| GPT-4.1 | **86.0** | **83.8** | 81.6 | **82.2** | **81.7** |
| DeepSeek-V3 | 82.8 | 79.5 | **81.9** | 80.8 | 79.9 |

**说明**: [[RewardAnything]] 在所有原则类别上均与 GPT-4.1 保持竞争力，在 Tone（语气）类别上略有优势。

### Table 4: 消融实验（来自论文 Table 4）

| 变体 | Chat | Code | Safety | Math | **Overall** | 说明 |
|------|------|------|--------|------|-------------|------|
| **Full RewardAnything-8B** | 81.6 | 81.9 | 84.4 | 79.6 | **81.9** | 完整模型 |
| Backbone (Qwen3-8B) | 71.7 | 69.2 | 77.7 | 66.6 | 71.3 | 无训练的基础模型 |
| **Training Ablations** | | | | | | |
| — Principles | 71.4 | 57.5 | 75.6 | 68.2 | 67.4 | 训练和推理时移除原则，模拟传统 RM |
| Listwise → Pairwise | 74.4 | 69.6 | 79.2 | 70.4 | 73.2 | 将 listwise 数据转为 pairwise 比较 |
| GRPO → SFT | 59.0 | 64.6 | 66.4 | 60.4 | 62.3 | 替换为[[SFT]]监督微调 |
| **GRPO Reward Ablations** | | | | | | |
| — Relative Preference | 79.1 | 77.3 | 80.4 | 75.2 | 78.2 | 移除相对偏好奖励 |
| — Format | 79.3 | 77.8 | 80.6 | 75.3 | 78.5 | 移除格式奖励 |
| **Inference Ablation** | | | | | | |
| — Reasoning | 74.8 | 74.1 | 77.4 | 65.3 | 73.9 | 推理时移除 CoT 推理 |

**关键发现**: 
- **GRPO → SFT** 是最大的性能下降因素（81.9→62.3，降 19.6 点），SFT 容易过拟合记忆化，[[RL]]训练对泛化至关重要
- **移除原则** 下降显著（81.9→67.4，降 14.5 点），说明明确的原则引导对模型遵循用户偏好至关重要
- **Listwise → Pairwise** 下降明显（81.9→73.2，降 8.7 点），说明 listwise 格式对多候选排序训练更有效
- 移除推理时[[Chain-of-Thought|思维链]]下降 8.0 点，说明推理能力对准确评估的重要性
- 相对偏好奖励和格式奖励各自贡献约 3-4 点

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RABENCH（论文自建） | 1,002 个偏好排名（31,806 偏好对） | 50 条原则，4 个领域（Chat/Code/Safety/Math），5 个原则类别 | 原则遵循能力评估 |
| [[RewardBench]] | 2,985 个测试问题 | 覆盖 Chat/Math/Code/Safety，分 Easy/Normal/Hard 三级难度 | 传统 RM 能力评估 |
| Skywork-Reward TrainSet（去污染后） | ~4,000 训练样本（去污染后） | 150 条原则，完全合成生成 | GRPL 训练 |

### 训练数据构建

- **原则来源**: 从 200 条原则池中采样 150 条作为训练用（与评估用的 50 条不重叠）
- **提示来源**: 经过去污染处理的 [[Skywork-Reward]] 训练集
- **回答生成**: 由 10 个不同 LLM（6 个模型家族）根据原则生成
- **标注**: 4 个最佳 LLM 评委 + [[Dynamic Programming|动态规划]] 共识算法
- **规模**: 约 4,000 个训练样本（等价于 173K 偏好对），完全合成无需人工标注

### 实现细节

- **Backbone**: [[Qwen3]]-8B
- **训练方法**: [[GRPO]]（Group Relative Policy Optimization）
- **奖励**: 格式奖励 $\lambda_f=0.15$ + 准确度奖励 $\lambda_a=0.85$
- **评估指标**: 准确率（Accuracy）、[[Kendall's Tau]]、[[NDCG]]
- **推理**: 使用 [[Chain-of-Thought]] 推理增强评估质量

### 可视化结果

论文未展示定性可视化案例（如具体的原则-评分示例），但在论文中通过 case study 展示了 RewardAnything 如何根据不同的原则（如"简洁优于详细"vs"详细优于简洁"）动态调整评分。

---

## 批判性思考

### 优点
1. **范式创新**: 将[[Instruction Following|指令遵循]]概念引入奖励模型，是一个重要的范式转变，使得 RM 可以像 LLM 一样动态适配用户需求
2. **全面评估**: 不仅提出新方法，还建立了专门评估原则遵循能力的 RABENCH 基准，推动了整个领域的发展
3. **实验扎实**: 在传统基准（[[RewardBench]]）上实现 SOTA，同时在原则遵循基准上媲美 GPT-4.1（仅 8B 参数），消融实验设计完整
4. **可解释性**: 结构化的推理+评分+排序输出提供了良好的可解释性
5. **实用价值**: 能与现有 [[PPO]]、[[GRPO]] 等 [[RLHF]] 流程无缝集成

### 局限性
1. **[[Length Bias]] 缓解深度有限**: 虽然 Hard 分项表现好，但论文未深入分析 RewardAnything 为何能缓解长度偏差，原理分析不足
2. **原则覆盖有限**: 50 条评估原则（150 条训练原则）虽多样化，但真实世界的评估标准远不止这些；对未见原则的真实泛化能力还需进一步检验
3. **合成训练数据**: 完全使用 LLM 合成数据（无人工验证），可能引入系统偏差；虽然评估数据有人工验证，但训练数据仅依赖 LLM-as-Judge 的自洽性
4. **计算开销**: 生成式推理+排序的计算成本高于传统判别式 RM（只需输出一个标量），在 [[RLHF]] 训练中可能成为瓶颈
5. **k 值限制**: 论文未讨论候选回答数量 $k$ 对性能的影响，实际应用中 $k$ 可能需要根据场景调整
6. **仅 8B 版本**: 仅有 Qwen3-8B 版本，更大尺度上的效果和扩展性未知

### 潜在改进方向
1. **扩展到更大模型**: 在 32B/70B+ 规模上验证方法有效性
2. **混合架构**: 将生成式推理与判别式评分结合，在保持精度的同时降低推理成本
3. **主动原则生成**: 让模型主动询问/生成最适合当前任务的原则
4. **多轮原则对话**: 允许多轮交互来精化和调整评估标准
5. **原则库共享**: 建立社区维护的原则库，方便用户开箱即用

### 可复现性评估
- [x] 代码开源（[GitHub](https://github.com/WisdomShell/RewardAnything)）
- [x] 预训练模型（[HuggingFace](https://huggingface.co/WisdomShell/RewardAnything-8B-v1)）
- [x] 训练细节完整（论文中有详细超参数设置）
- [ ] 数据集可获取（RABENCH 数据集需进一步确认是否公开）

---

## 关联笔记

### 基于
- [[GRPO]]: [[DeepSeekMath]] 提出的 Group Relative Policy Optimization，本方法的核心训练算法
- [[RewardBench]]: 传统奖励模型评估基准，[[RABENCH]] 的提示来源
- [[Qwen3]]: RewardAnything 的基础模型

### 对比
- [[RM-R1]]: 基于推理能力的奖励模型方法，本方法的主要基线之一
- [[Skywork-Reward]]: 基于 Llama 的奖励模型，本方法的基线之一
- [[LLM-as-Judge]]: 传统的 LLM 评判范式，本方法在原理上不同（专用 RM vs 通用 LLM）

### 方法相关
- [[RLHF]]: 强化学习人类反馈，奖励模型的核心应用场景
- [[PPO]]: 近端策略优化，[[GRPO]] 的基础
- [[Reward Model|奖励模型 (Reward Model)]]: 本方法的核心研究对象
- [[PRM|Process Reward Model]]: 过程奖励模型，当前奖励模型的另一重要方向

### 评估相关
- [[RABENCH]]: 本论文提出的原则遵循奖励模型基准
- [[Kendall's Tau]]: 排序一致性评估指标
- [[NDCG]]: 归一化折损累计增益，排序质量评估指标
- [[Length Bias]]: 奖励模型的常见偏差，本方法尝试缓解的问题

---

## 速查卡片

> [!summary] RewardAnything: Generalizable Principle-Following Reward Models
> - **核心**: 提出可遵循自然语言原则的奖励模型，在推理时动态适配评估标准
> - **方法**: 基于 [[Qwen3]]-8B + [[GRPL]]（[[GRPO]] 强化学习），输入原则+提示+回答，输出推理+评分+排序
> - **结果**: [[RewardBench]] SOTA 86.4%，[[RABENCH]] 81.9%（可媲美 GPT-4.1）
> - **代码**: https://github.com/WisdomShell/RewardAnything

---

*笔记创建时间: 2026-07-14 20:30*