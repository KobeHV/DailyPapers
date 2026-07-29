---
title: "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains"
method_name: "RaR"
authors: [Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, Sean Hendryx]
year: 2025
venue: arXiv
tags: [reinforcement-learning, rubric-evaluation, llm-as-judge, grpo, reward-modeling, post-training, medical-ai]
zotero_collection: ""
image_source: local
arxiv_html: "https://arxiv.org/html/2507.17746"
created: 2025-07-29
---

# 论文笔记：Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Scale AI |
| 日期 | July 2025 |
| 项目主页 | — |
| 对比基线 | [[DeepSeekMath|GRPO]] (原版), Direct-Likert, Reference-Likert |
| 链接 | [arXiv](https://arxiv.org/abs/2507.17746) / [RaR-Medicine](https://huggingface.co/datasets/anisha2102/RaR-Medicine) / [RaR-Science](https://huggingface.co/datasets/anisha2102/RaR-Science) |

---

## 一句话总结

> 将实例级评分量表（Rubric）转化为 [[GRPO]] [[On-Policy RL]] 的结构化奖励信号，使 [[RLVR]] 突破可验证领域（数学/代码），扩展到医学和科学等需要多维度主观判断的真实推理任务。

---

## 核心贡献

1. **RaR 框架**: 提出 Rubrics as Rewards，将 checklist 风格的评分标准转化为 [[On-Policy RL]] 的奖励函数，作为 [[RLVR]]（二元验证）和 [[RLHF]]（偏好排序）之间的中间地带。
2. **RaR-Medicine 与 RaR-Science 数据集**: 分别用 GPT-4o 和 o3-mini 合成 20k 条带实例级评分量表的医学/科学推理训练数据，并公开发布。
3. **奖励聚合策略**: 系统对比 Explicit Aggregation（加权求和）和 Implicit Aggregation（LLM 整体打分）两种评分聚合方式。
4. **跨评估格式泛化**: RaR 训练的模型在 rubric 评分和多项选择题两种评估格式上均表现出色，[[GPQA-Diamond]] 上相对提升 7%，证明 rubric 引导的技能可以泛化。
5. **降低 Judge 规模依赖**: Rubric 引导使小型 [[LLM-as-Judge]] 的对齐准确率接近大型 judge，缩小了规模差距。

---

## 问题背景

### 要解决的问题

[[RLVR]]（使用可验证奖励的强化学习）在数学和代码等有明确二元对错信号的领域取得了巨大成功（如 [[DeepSeekMath|DeepSeek-R1]]）。然而，**如何将 RLVR 扩展到医学、科学等没有单一正确答案的真实推理任务**仍是一个核心挑战。这些领域的评估依赖多维度、主观的判断（准确性、完整性、安全性、同理心等），而非简单的对错。

### 现有方法的局限

1. **偏好奖励模型**: [[RLHF]] 中使用的偏好模型容易过拟合到表面特征（回复长度、格式偏好、标注者偏见），且需要大量人工 pairwise 比较。
2. **直接 Likert 评分**: [[LLM-as-Judge]] 直接给回复打 1-10 分（Direct-Likert），缺乏结构化指导，评分一致性差，尤其是小 judge 模型。
3. **参考答案评分**: 虽然参考答案提供了基准，但转换为标量奖励信号时丢失了大量细粒度信息。
4. **评分量表仅用于评估**: [[HealthBench]] 等工作中，评分量表仅用于评估阶段，其在 on-policy 训练中作为奖励信号的潜力未被探索。

### 本文的动机

评分量表（Rubric）天然适合作为奖励信号——它将"什么是好的回复"分解为模块化、可解释的子目标，提供了一个介于二元验证和粗粒度偏好排序之间的**中间地带**。将评分量表从评估工具转化为训练奖励，可以闭合"rubric-to-learning"循环。

---

## 方法详解

### 整体框架

RaR (Rubrics as Rewards) 包含两个阶段：

- **阶段 1 — Rubric Generation**: 使用强 LLM（GPT-4o / o3-mini）根据参考答案为每个 prompt 生成 7-20 条自包含的评分标准，每条标注重要性权重（Essential / Important / Optional / Pitfall）。
- **阶段 2 — GRPO Training**: 将这些评分标准作为 [[LLM-as-Judge]] 的 prompt 来估计奖励，驱动 [[GRPO]] on-policy 策略优化。

![[RaR_fig1_overview.png]]

### 问题形式化

设 $x$ 为输入 prompt，$\hat{y} \sim \pi_\theta(\cdot \mid x)$ 为模型采样的回复。

每个 prompt $x$ 关联 $k$ 条评分标准 $\{(w_j, c_j)\}_{j=1}^k$：
- $w_j \in \mathbb{R}$：标准 $j$ 的权重
- $c_j: (x, \hat{y}) \rightarrow \{0, 1\}$：二元正确性函数

### 奖励聚合策略

#### Explicit Aggregation（显式聚合）

每条评分标准由 [[LLM-as-Judge]] 独立评估，最终奖励为加权归一化求和：

$$
r(x, \hat{y}) = \frac{\sum_{j=1}^{k} w_j \cdot c_j(x, \hat{y})}{\sum_{j=1}^{k} w_j}
$$

归一化使不同数量评分标准的 prompt 之间奖励可比。实验中采用二元检查 $c_j$，但可扩展为连续值。

权重映射：{"Essential": 1.0, "Important": 0.7, "Optional": 0.3, "Pitfall": 0.9}（Pitfall 以正面形式表述，"避免错误信息"，满足则贡献正分）。

#### Implicit Aggregation（隐式聚合）

将所有评分标准和分类权重直接传给 [[LLM-as-Judge]]，让模型自身完成聚合，输出 1-10 的 [[Likert Scale]] 评分（归一化到 [0,1]）：

$$
r_{\text{implicit}}(x, \hat{y}) = f(x, \hat{y}, \{d_j\}_{j=1}^k)
$$

这种方式避免了手动调权的困难，是实验中表现最好的方法。

### RaR 是 RLVR 的超集

Rubric-based RL 可自然退化到标准 [[RLVR]]：当 $k=1, w_1=1$，且 $c_1(x, \hat{y})$ 退化为单一可验证正确性函数时：

$$
r_{\text{RLVR}}(x, \hat{y}) = \operatorname{match}(y, \hat{y})
$$

---

## 关键公式

### 公式 1: [[Rubric-based Evaluation|Explicit Reward Aggregation]]

$$
r(x, \hat{y}) = \frac{\sum_{j=1}^{k} w_j \cdot c_j(x, \hat{y})}{\sum_{j=1}^{k} w_j}
$$

**含义**: 将 $k$ 条评分标准的独立判断加权求和并归一化，确保跨 prompt 可比。

**符号说明**:
- $x$: 输入 prompt
- $\hat{y}$: 模型生成的回复
- $w_j \in \mathbb{R}$: 第 $j$ 条评分标准的权重
- $c_j(x, \hat{y}) \in \{0, 1\}$: 二元正确性判断函数
- $k$: 该 prompt 的评分标准总数

### 公式 2: [[Rubric-based Evaluation|Implicit Reward Aggregation]]

$$
r_{\text{implicit}}(x, \hat{y}) = f(x, \hat{y}, \{d_j\}_{j=1}^k)
$$

**含义**: 将评分标准和回复全部交给 LLM judge 做整体评分，输出单一 [[Likert Scale]] 分数。

**符号说明**:
- $f$: LLM-based judge 函数
- $\{d_j\}_{j=1}^k$: 所有评分标准集合
- 输出: 1-10 Likert 评分，归一化到 [0, 1]

### 公式 3: [[RLVR|RLVR 作为 RaR 的特例]]

$$
r_{\text{RLVR}}(x, \hat{y}) = \operatorname{match}(y, \hat{y})
$$

**含义**: 当评分标准退化为单一可验证正确性条件时，RaR 等价于标准 RLVR。

**符号说明**:
- $y$: 正确答案
- $\operatorname{match}(y, \hat{y}) \in \{0, 1\}$: 可验证正确性条件（精确匹配或测试用例通过）

---

## 关键图表

### Figure 1: Overview / 系统概览

![[RaR_fig1_overview.png]]

**说明**: RaR 框架的两阶段流程。(i) Rubric Generation: 使用强 LLM + 参考答案，基于四个设计原则（Expert Grounding、Comprehensive Coverage、Criterion Importance、Self-Contained Evaluation）为每个 prompt 生成评分标准。(ii) GRPO Training: 评分标准驱动 [[LLM-as-Judge]] 估计奖励，通过 [[GRPO]] on-policy 循环优化策略模型。

### Figure 2: Experiment Results / 实验结果

![[RaR_fig2_results.png]]

**说明**: 左：[[HealthBench]] 上的分轴评分（Communication quality, Accuracy, Completeness, Instruction following, Context awareness + Overall）。RaR-Implicit 达到 31.2% overall，远超 Direct-Likert (7.7%) 和 Reference-Likert (25.5%)。右：[[GPQA-Diamond]] 上的 10 次运行均值准确率，RaR-Implicit (37.6%) 优于 Direct-Likert (35.0%) 和 Reference-Likert (36.5%)。

### Figure 3: Alignment Study / 对齐研究

![[RaR_fig3_alignment.png]]

**说明**: 不同规模的 [[LLM-as-Judge]] (GPT-4o-mini, Qwen-7B/14B/32B-Instruct) 在使用 rubric 引导（橙色）vs 直接 Likert（蓝色）评分时的 pairwise 偏好准确率。Rubric 引导在所有 judge 规模上均提升准确率，对小模型的增益最大，缩小了与大模型的差距。绿色线显示无参考答案的纯合成评分量表性能，明显低于有参考答案引导的版本，证明**专家信号（参考答案）对评分量表质量至关重要**。

---

## 实验

### 数据集

| 数据集 | 规模 | 评分量表生成模型 | 领域 | 用途 |
|--------|------|-----------------|------|------|
| RaR-Medicine | 20,166 prompts | GPT-4o | 医学诊断/治疗/知识/伦理等 | 训练 |
| RaR-Science | 20,625 prompts | o3-mini | 化学/物理/生物/量子力学等 | 训练 |
| HealthBench | 5,000 (1k 消融) | 医师撰写 | 临床对话安全与有用性 | 评估（Rubric-based） |
| GPQA-Diamond | — | — | 科学多项选择题 | 评估（Multiple-Choice） |

### 实现细节

- **Base Policy**: Qwen2.5-7B
- **算法**: [[GRPO]] (Group Relative Policy Optimization)
- **Judge Model**: gpt-4o-mini（奖励估计）
- **Batch Size**: 96（effective）
- **Rollouts per Prompt**: $k = 16$
- **Learning Rate**: $5 \times 10^{-6}$，constant schedule with 10% linear warmup
- **Max Length**: 3584 tokens
- **Sampling Temperature**: 1.0
- **Num Train Steps**: 300
- **硬件**: 单节点 8x NVIDIA H100 GPU

### Baseline 方法

| 方法 | 描述 |
|------|------|
| **Qwen2.5-7B** | Base policy，未训练 |
| **Qwen2.5-7B-Instruct** | 官方指令微调版本 |
| **Direct-Likert** | [[LLM-as-Judge]] 直接打 1-10 分作为奖励（无评分量表） |
| **Reference-Likert** | Judge 对比参考答案后打 1-10 分 |
| **RaR-Predefined** | 使用固定的通用评分量表（非实例级），显式聚合 |
| **RaR-Explicit** | 使用实例级评分量表 + 显式加权求和 |
| **RaR-Implicit** | 使用实例级评分量表 + LLM 隐式整体评分 |

### 主要结果

#### HealthBench (Rubric-Based 评估，自由文本回复)

| 方法 | Overall Score |
|------|-------------|
| Qwen2.5-7B | 22.7% |
| Qwen2.5-7B-Instruct | 28.9% |
| Direct-Likert | 7.7% |
| Reference-Likert | 25.5% |
| RaR-Predefined | 12.5% |
| RaR-Explicit | 29.7% |
| **RaR-Implicit** | **31.2%** |

**关键发现**: RaR-Implicit 相对 Direct-Likert 提升 **31%**（相对），超越 Reference-Likert。Direct-Likert 表现极差（7.7%），说明无结构化的直接评分作为奖励信号可能导致 [[Reward Hacking]]。

#### GPQA-Diamond (多项选择题)

| 方法 | Mean Accuracy (95% CI) |
|------|----------------------|
| Qwen2.5-7B | 31.7% |
| Qwen2.5-7B-Instruct | 35.0% |
| Direct-Likert | 34.8% |
| Reference-Likert | 36.5% |
| RaR-Predefined | 31.7% |
| RaR-Explicit | 36.9% |
| **RaR-Implicit** | **37.6%** |

**关键发现**: RaR-Implicit 在多项选择题上也最优（相对 Direct-Likert 提升 7%），说明 rubric 训练出的技能可以**泛化到不同评估格式**。

### 消融实验

#### 评分量表生成方式的影响（Table 1，HealthBench-1k 子集）

| 训练方法 | Overall Score |
|---------|-------------|
| Expert-Answer-SFT | 20.4% |
| Simple-Likert | 23.9% |
| Reference-Likert | 31.7% |
| RaR-Implicit-Synthetic-NoRef | 32.0% |
| **RaR-Implicit-Synthetic** | **35.9%** |
| RaR-Implicit-Human | 34.8% |

**关键发现**: 合成评分量表配合参考答案（35.9%）已达人类撰写评分量表（34.8%）的水平。无参考答案的纯合成评分量表（32.0%）显著下降，证明**专家信号对评分量表质量的决定性作用**。

#### 评分量表设计元素（Table 2，HealthBench-1k）

| 消融设置 | Overall Score |
|---------|-------------|
| Essential-Only Rubrics | 34.9% |
| No Categorical Labels | 38.8% |
| No Pitfall Criteria | 37.2% |
| All Rubrics | 37.2% |

**关键发现**: 只保留 Essential 标准会导致信息不足（34.9%）；去掉分类权重标签反而略微提升（38.8% vs 37.2%），说明手动权重可能引入偏差。

#### 评分量表生成模型的影响（Table 3，HealthBench-1k，无参考答案）

| Rubric 生成模型 | Overall Score |
|----------------|-------------|
| O3-mini (w/ reference) | 35.9% |
| GPT-4o | 34.2% |
| GPT-4o-mini | 32.7% |
| Qwen-72B-Instruct | 32.7% |
| O3-mini | 32.4% |
| Qwen-32B-Instruct | 31.1% |
| Qwen-7B-Instruct | 31.9% |

**关键发现**: 更强的生成模型 $\rightarrow$ 更有效的评分量表。GPT-4o 在无参考条件下最优。但**所有无参考模型均远低于有参考引导的 o3-mini**（35.9%），再次证明专家信号的重要性。

#### Judge 规模与评分方式对比（Table 11，synthetic medical data）

| Judge Model | RaR-Implicit | Direct-Likert |
|------------|-------------|--------------|
| GPT-4o-mini | 27.9% | 25.3% |
| Qwen-32B-Instruct | 26.2% | 25.4% |
| Qwen-14B-Instruct | 25.0% | 24.9% |
| Qwen-7B-Instruct | 26.7% | 22.0% |

**关键发现**: Rubric 引导在小 judge 上的增益最大（Qwen-7B: +4.7%），且 rubric-based 分数聚合度更高（0.250-0.279 vs 0.220-0.254），说明结构化标准降低了 judge 模型的能力门槛。

### 附录：数据集统计

#### Table 4: RaR-Medicine 聚合统计

| Metric | Value |
|--------|-------|
| Total examples | 20,166 |
| Avg. rubrics per question | 7.5 |
| Avg. question length (words) | 45.0 |

#### Table 5: RaR-Medicine 评分标准类型分布

| Rubric Type | Count | Percent |
|------------|-------|---------|
| Important | 52,748 | 34.1% |
| Essential | 47,584 | 30.7% |
| Optional | 34,261 | 22.1% |
| Pitfall | 20,215 | 13.1% |

#### Table 6: RaR-Medicine 医学主题分布

| Topic | Count | Percent |
|-------|-------|---------|
| Medical Diagnosis | 10,147 | 50.3% |
| Medical Treatment | 3,235 | 16.0% |
| Medical Knowledge | 2,557 | 12.7% |
| Medical Diag. and Mngmnt | 2,033 | 10.1% |
| Medical Biology | ~770 | 3.8% |
| Medical Ethics | 428 | 1.9% |
| Health Physics | 377 | 1.4% |
| Epidemiology & Pub. Health | 276 | 1.1% |
| General Medicine | 216 | 0.6% |
| Forensic Medicine | 113 | 0.1% |

#### Table 7: RaR-Science 聚合统计

| Metric | Value |
|--------|-------|
| Total examples | 20,625 |
| Avg. rubrics per question | 7.5 |
| Avg. question length (words) | 52.6 |

#### Table 8: RaR-Science 评分标准类型分布

| Rubric Type | Count | Percent |
|------------|-------|---------|
| Important | 52,315 | 34.8% |
| Essential | 42,739 | 28.4% |
| Optional | 33,622 | 22.3% |
| Pitfall | 21,808 | 14.5% |

#### Table 9: RaR-Science 科学主题分布

| Topic | Count | Percent |
|-------|-------|---------|
| General Chemistry | 3163 | 15.3% |
| Quantum Mechanics | 3158 | 15.3% |
| Physical Chemistry | 2761 | 13.4% |
| Statistical Mechanics | 2530 | 12.3% |
| Organic Chemistry | 2059 | 10.0% |
| General Physics | 1439 | 7.0% |
| Condensed Matter Physics | 1387 | 6.7% |
| Genetics | 1378 | 6.7% |
| Molecular Biology | 1378 | 6.7% |
| Astrophysics | 815 | 4.0% |
| Inorganic Chemistry | 409 | 2.0% |
| Analytical Chemistry | 407 | 2.0% |
| Electromagnetism | 398 | 1.9% |
| Optics | 239 | 1.2% |
| High Energy Physics | 143 | 0.7% |

#### Table 10: GRPO 训练超参数

| Hyperparameter | Value |
|---------------|-------|
| num_rollouts_per_prompt | 16 |
| batch_size (effective) | 96 |
| sampling_temperature | 1.0 |
| warmup_ratio | 0.1 |
| learning_rate | 5.0e-06 |
| lr_scheduler_type | constant_with_warmup |
| max_length | 3584 |
| num_train_steps | 300 |

---

## 批判性思考

### 优点

1. **概念简洁优雅**: RaR 将一个自然的想法（评分量表作为奖励）系统化实现，提供了 RLVR 和 RLHF 之间的平滑过渡，数学形式化清晰（RLVR 是 RaR 的特例）。
2. **实验设计全面**: 涵盖两种聚合策略、两种领域（医学+科学）、两种评估格式（rubric-based + multiple-choice）、多种消融（生成模型、设计元素、judge 规模），对比充分。
3. **实用性强**: 公开了两个 20k 规模的数据集，训练成本可控（7B 模型 + 单节点 8xH100 + 300 steps），可复现性高。
4. **关键洞察有价值**: 发现 expert grounding（参考答案）对评分量表质量至关重要、rubric 降低 judge 规模依赖、结构化标准帮助小 judge 对齐——这些对后续工作有指导意义。

### 局限性

1. **领域范围有限**: 仅验证了医学和科学两个领域，论文也承认需要在对话、工具使用、agentic 任务等更广泛场景中验证。
2. **评分量表合成依赖强 LLM**: 需要 GPT-4o/o3-mini 级别模型 + 参考答案，对资源受限场景不够友好。纯合成（无参考答案）评分量表的性能仍有较大下降。
3. **仅测试两种聚合策略**: Explicit 和 Implicit 是两个极端（完全手动 vs 完全黑箱），更高级的策略（学习权重、动态权重、课程权重）未被探索。
4. **Pitfall 评分标准的合成质量不足**: 消融显示去掉 Pitfall 后性能几乎没变（37.2% vs 37.2%），说明合成的负例标准不够具体或不够相关——这可能是合成方法的根本局限。
5. **Base 模型固定**: 只在 Qwen2.5-7B 上实验，未验证方法在不同规模/系列模型上的泛化性。
6. **未探索与 Process Reward Model 的结合**: 评分量表目前用于评估最终输出，是否能扩展到过程监督（step-level rubrics）未讨论。

### 潜在改进方向

1. **动态权重学习**: 训练过程中根据模型弱点自动调整各标准的权重（curriculum learning），初期重 Essential 后期重 Optional。
2. **评分量表 + Process Reward 融合**: 将 rubric 标准分解为 process-level checklist，提供更密集的训练信号。
3. **多轮迭代**: 用当前训练出的 policy 生成新回复，重新合成更具针对性的评分量表（adversarial rubric generation）。
4. **跨领域迁移**: 验证在医学领域训练的 rubric-guided policy 是否能在科学领域 zero-shot 受益。

### 可复现性评估

- [x] 代码开源（提到使用开源项目的内部平台 RLXF，但未提供公开代码 repo——**部分可复现**）
- [ ] 预训练模型
- [x] 训练细节完整（所有超参数在 Table 10）
- [x] 数据集可获取（RaR-Medicine + RaR-Science 在 HuggingFace 公开）

---

## 关联笔记

### 基于
- [[GRPO]]: RaR 使用 GRPO 作为底层 on-policy RL 算法
- [[RLVR]]: RaR 从 RLVR 出发，将其推广到不可验证领域
- [[DeepSeekMath]]: GRPO 算法的引入者，RaR 延续其 on-policy 训练范式
- [[HealthBench]]: RaR 采用其医师撰写的评分量表作为评估基准

### 对比
- [[RLHF]]: RaR 作为 RLHF（偏好排序）的替代方案，用结构化评分标准代替 pairwise 比较
- [[DPO|Direct Preference Optimization]]: 偏好优化方法，RaR 关注的是评分标准转为奖励信号
- [[Reward Hacking]]: Direct-Likert 低至 7.7% 暗示了 reward hacking 问题，RaR 通过结构化评分标准缓解

### 方法相关
- [[On-Policy RL]]: RaR 核心训练范式
- [[LLM-as-Judge]]: RaR 的奖励信号来源，也是重点研究 judge 规模影响的主题
- [[Rubric-based Evaluation]]: RaR 的核心创新——将评分量表从评估工具转为训练信号
- [[Likert Scale]]: Implicit Aggregation 的输出形式
- [[Reward Shaping]]: 评分量表本质上是一种 reward shaping 方法

### 硬件/数据相关
- [[Qwen2.5]]: Base policy 模型
- [[GPQA-Diamond]]: 科学多项选择评估基准

---

## 速查卡片

> [!summary] Rubrics as Rewards (RaR)
> - **核心**: 将实例级评分量表转化为 GRPO 的结构化奖励，使 RLVR 突破二元验证限制
> - **方法**: LLM 合成评分量表 + Explicit/Implicit 聚合 + GRPO on-policy 训练
> - **结果**: HealthBench 相对提升 31%, GPQA-Diamond 提升 7%; rubric 降低小 judge 的规模依赖
> - **数据**: [RaR-Medicine](https://huggingface.co/datasets/anisha2102/RaR-Medicine) / [RaR-Science](https://huggingface.co/datasets/anisha2102/RaR-Science)

---

*笔记创建时间: 2025-07-29*
