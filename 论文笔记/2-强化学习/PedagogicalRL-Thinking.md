---
title: "Rewarding How Models Think Pedagogically: Integrating Pedagogical Reasoning and Thinking Rewards for LLMs in Education"
method_name: "PedagogicalRL-Thinking"
authors: [Unggi Lee, Jiyeong Bae, Jaehyeon Park, Haeun Park, Taejun Park, Younghoon Jeon, Sungmin Cho, Junbo Koh, Yeil Jeong, Gyeonggeon Lee]
year: 2026
venue: arXiv
tags: [reinforcement-learning, pedagogical-alignment, llm-tutor, grpo, thinking-reward, reasoning-llm, education]
zotero_collection: 2-强化学习
image_source: online
arxiv_html: https://arxiv.org/html/2601.14560
created: 2026-08-01
---

# 论文笔记：Rewarding How Models Think Pedagogically

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Chosun University, Korea University 等 |
| 日期 | January 2026 |
| 项目主页 | — |
| 对比基线 | [[PedagogicalRL]], [[DeepSeek-R1]], Qwen2.5-7B |
| 链接 | [arXiv](https://arxiv.org/abs/2601.14560) / [Code](https://anonymous.4open.science/repo/pedagogical_reasoning_submission-2787/) |

---

## 一句话总结

> 首次将强化学习的奖励信号扩展到 LLM 推理阶段的"思考过程"，结合教育理论引导的思维提示，训练出更好的 AI 数学导师。

---

## 核心贡献

1. **[[Thinking Reward|思考奖励]]机制**: 首次将 RL 奖励信号从可见回复扩展到模型内部 `<think>` 推理轨迹，用 LLM Judge 评估思考过程的教学质量
2. **[[Polya's Problem-Solving Framework|教学推理提示]]**: 基于 Polya 四步解题框架设计领域特定的思维引导提示，显著优于通用提示
3. **思考+提示协同效应**: 实验证明思考奖励只在结合教学提示时才发挥最大效果，二者存在协同增强关系
4. **分布外泛化**: 仅在数学辅导对话上训练的模型，在未见过的教育基准测试（WBEB）上展现出显著的泛化能力

---

## 问题背景

### 要解决的问题

大语言模型（[[Intelligent Tutoring System|LLM 智能辅导系统]]）在教育中的应用日益广泛，但现有 RL 训练方法只优化模型的**可见输出**（visible response），忽略了模型的**内部推理过程**（thinking process）。这相当于只考核学生的"答卷"而不关注其"草稿"——草稿中的推理质量直接影响最终输出的教学质量。

### 现有方法的局限

- [[PedagogicalRL]]（Dinucu-Jianu et al., 2025）首次将 [[GRPO|RL 训练]] 应用于 LLM 导师，但仅基于可见回复计算奖励（答案泄露检测 + 有帮助性评估）
- 现有推理增强方法（如 [[DeepSeek-R1]] 的冷启动训练）仅关注数学/代码推理的正确性，不涉及教学推理质量
- 教育领域的 LLM 对齐缺乏对**教学思维过程**的系统评估和优化

### 本文的动机

教学是一场"思维的对话"——优秀的人类教师不仅在嘴上引导学生，更在大脑中持续进行教学推理（判断学生理解程度、选择合适策略、调整解释方式）。让 LLM 导师也具备这种"教学思维"，并通过 RL 直接优化这种思维的质量，是提升 AI 辅导效果的关键路径。

---

## 方法详解

### 模型架构

PedagogicalRL-Thinking 将 LLM 辅导对话建模为 [[Markov Decision Process|MDP]]，在 [[GRPO]] 框架下优化：

- **状态 $s_t$**: 对话历史（学生消息 + 导师回复）
- **动作 $a_t$**: 导师的思考轨迹 $a_t^{\text{think}}$（`<think>` 标签内）+ 可见回复 $a_t^{\text{visible}}$
- **Base Model**: [[DeepSeek-R1|DeepSeek-R1-0528-Qwen3-8B]]（思考条件）/ Qwen2.5-7B（NoThink 基线）
- **训练算法**: [[GRPO]]（通过 [[veRL]] 框架实现），无需单独的价值模型
- **Student Simulator**: Llama-3.1-8B-Instruct

### 核心模块

#### 模块1: [[Polya's Problem-Solving Framework|教学推理提示（Pedagogical Reasoning Prompting）]]

**设计动机**: 利用 Polya 的教育理论框架指导 LLM 的内部思考，使其推理过程遵循成熟的教学方法论，而非依靠模型自行摸索

**具体实现**:
- 在 System Prompt 中注入 Polya 四步框架：理解问题 → 制定计划 → 逐步执行 → 回顾验证
- 指导模型在 `<think>` 阶段按此流程组织推理，而非通用的"一步一步思考"
- 提示示例：*"Guide the student by helping them understand the problem, devise a plan, carry out the plan step by step, and verify whether the answer makes sense."*

#### 模块2: [[Thinking Reward|思考奖励（Thinking Reward）]]

**设计动机**: 将 RL 奖励信号从"答卷质量"扩展到"草稿质量"，直接用教学标准评估模型的思考过程

**具体实现**:
- 使用 GPT-4o-mini 作为思考质量 Judge，评估 `<think>` 轨迹的三个维度：
  - 推理的教学适当性（Pedagogical appropriateness）
  - 对学生理解的考量（Consideration of student understanding）
  - 元认知意识（Metacognitive awareness in planning responses）
- 思考奖励分数 $r_{\text{think}} \in [0, 1]$

---

## 关键公式

### 公式1: [[Composite Reward|复合奖励函数]]

$$
r = r_{\text{sol}} + (r_{\text{ped}} - 1.0) \cdot \lambda_{\text{ped}} + (r_{\text{think}} - \theta) \cdot \lambda_{\text{think}}
$$

**含义**: 三部分奖励的加权组合 —— 解题效果 + 教学规范 + 思考质量

**符号说明**:
- $r_{\text{sol}}$: 解题正确率奖励（学生被辅导后 $K=8$ 次尝试的平均正确率）
- $r_{\text{ped}}$: 教学质量奖励，由两个 [[LLM Judge]]（Leak Judge + Help Judge）评估，两者都通过则为 1.0，否则为 0.0
- $r_{\text{think}}$: 思考质量奖励分数 $\in [0, 1]$
- $\lambda_{\text{ped}} = 0.75$: 教学质量权重
- $\lambda_{\text{think}} = 0.3$: 思考奖励权重
- $\theta = 0.6$: 思考奖励阈值（低于阈值时为负奖励）

### 公式2: [[GRPO|GRPO 优势函数]]

$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{(s,a) \sim \pi_{\text{old}}} \left[ \min\left( \rho(\theta) A, \operatorname{clip}(\rho(\theta), 1-\epsilon, 1+\epsilon) A \right) \right]
$$

**含义**: GRPO 通过组内采样比较计算优势，避免训练额外的价值模型，配合 veRL 框架高效实现

**符号说明**:
- $\rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$: 策略比率
- $A$: 基于组内奖励标准化计算的优势
- $\epsilon$: clip 超参数

### 公式3: [[Student Simulator|学生解题正确率]]

$$
r_{\text{sol}} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}[\,\text{student solves correctly on attempt } k\,]
$$

**含义**: 辅导效果的客观度量 —— 学生被辅导后在 $K$ 次独立尝试中的平均正确率

**符号说明**:
- $K = 8$: 每题的独立尝试次数
- Student Model: Llama-3.1-8B-Instruct，以数学学生角色被 prompt

---

## 关键图表

### Figure 1: Framework Overview / 框架概览

![Figure 1](https://arxiv.org/html/2601.14560/x1.png)

*注：图片来自 arXiv HTML 源，如无法加载可访问 [arXiv HTML](https://arxiv.org/html/2601.14560) 查看*

**说明**: 框架概览。输入为数学问题 + 学生状态，导师模型在 `<think>` 阶段进行教学推理（受 Polya 框架引导），在可见阶段输出辅导回复。三部分奖励 —— $r_{\text{sol}}$（学生解题效果）、$r_{\text{ped}}$（Leak/Help 教学规范）、$r_{\text{think}}$（思考教学质量）—— 共同驱动 GRPO 训练。

### Figure 2: Reasoning Trace Analysis / 推理轨迹分析

![Figure 2](https://arxiv.org/html/2601.14560/x2.png)

*注：图片来自 arXiv HTML 源，如无法加载可访问 [arXiv HTML](https://arxiv.org/html/2601.14560) 查看*

**说明**: 教学推理轨迹的定性分析。使用 Schoenfeld 数学教学框架（Explore, General 等类别）对各条件下的 `<think>` 轨迹进行编码。Ped. Think Reward 条件展现出最低的 Explore 比率（0.75%）和最高的 General 比率（81.02%），表明推理更聚焦、更结构化，避免了不必要的探索性发散。

### Figure 3: Experimental Results / 实验结果

![Figure 3](https://arxiv.org/html/2601.14560/x3.png)

*注：图片来自 arXiv HTML 源，如无法加载可访问 [arXiv HTML](https://arxiv.org/html/2601.14560) 查看*

**说明**: 核心实验结果对比。Ped. Think Reward（教学提示 + 思考奖励）在所有指标上均取得最优结果 —— 最高 Delta Solve（0.294）、最低 Leak Rate（0.172）、最高 Helpful Rate（0.776）。同时展示了思维启用带来的巨大提升（NoThink → Think NoReward: +134% Delta Solve, +306% Helpful）。

### Table 1: Main Results / 主要实验结果

| Condition | Base Model | Think | Prompt | Think Reward | **Delta Solve $\uparrow$** | **Leak $\downarrow$** | **Helpful $\uparrow$** |
|-----------|-----------|------|--------|-------------|:------:|:-----:|:------:|
| NoThink | Qwen2.5-7B | No | Normal | — | 0.120 | 0.300 | 0.180 |
| Think NoReward | Qwen3-8B | Yes | Normal | No | 0.281 | 0.180 | 0.730 |
| Think Reward | Qwen3-8B | Yes | Normal | Yes | 0.284 | 0.182 | 0.764 |
| Ped. Think NoReward | Qwen3-8B | Yes | Ped. | No | 0.275 | 0.214 | 0.766 |
| **Ped. Think Reward** | Qwen3-8B | **Yes** | **Ped.** | **Yes** | **0.294** | **0.172** | **0.776** |

**关键发现**:
- 启用思考（Think）带来巨大跃升：Delta Solve +134%，Helpful +306%
- 思考奖励（Think Reward）在普通提示下效果有限（0.281 → 0.284），但在教学提示下作用显著（0.275 → 0.294，即 +6.9%）
- 单独使用教学提示但不加思考奖励，Leak Rate 反而上升（0.214），说明需要奖励信号来抑制不当行为

### Table 2: Out-of-Distribution Generalization / WBEB 基准泛化

| Model | Subject Know. $\uparrow$ | Pedagogy Know. $\uparrow$ | Essay Scoring $\uparrow$ | Decision Making $\uparrow$ |
|-------|:------:|:------:|:------:|:------:|
| Qwen3-8B (Original) | 23.2 | 21.1 | 10.7 | 15.8 |
| Think NoReward | 21.9 (-1.3) | 27.6 (+6.5) | 23.7 (+13.0) | 54.4 (+38.6) |
| Think Reward | 21.7 (-1.5) | 29.5 (+8.4) | 23.6 (+12.9) | 53.4 (+37.6) |
| Ped. Think NoReward | 22.4 (-0.8) | 29.0 (+7.9) | 23.7 (+13.0) | 53.1 (+37.3) |
| **Ped. Think Reward** | **22.5 (-0.7)** | 26.0 (+4.9) | **25.5 (+14.8)** | 53.8 (+38.0) |

**关键发现**:
- 仅在数学辅导对话上训练的模型，在所有 WBEB 子维度上均展现出显著的分布外泛化
- Ped. Think Reward 在最小化 Subject Knowledge 损失（-0.7%）的同时，取得了最高的 Essay Scoring（25.5%）
- Teacher Decision Making 提升最大（+37~39 个百分点），表明教学推理能力发生了系统性迁移

### Table 3: Response Characteristics / 响应特征分析

| Condition | Visible Words | Think Words | Total Words | Unique Words | Math% (Think) | Math% (Visible) |
|-----------|:-----:|:-----:|:-----:|:-----:|:----:|:----:|
| NoThink | 98.18 | — | 98.18 | 60.38 | — | 27–30% |
| Think NoReward | 74.67 | 153.96 | 112.52 | 61.87 | ~37% | 27–30% |
| Think Reward | 72.91 | 157.16 | 114.02 | 62.18 | ~37% | 27–30% |
| Ped. Think NoReward | 95.05 | 185.59 | **141.30** | **71.57** | ~37% | 27–30% |
| Ped. Think Reward | 91.60 | 183.62 | 138.31 | 70.92 | ~37% | **29.76%** |

**关键发现**:
- 教学提示条件产生更长、词汇更丰富的回复（Total Words 141 vs 114，Unique Words 72 vs 62）
- Think 阶段的数学内容占比（~37%）高于可见回复（27–30%），说明思考阶段更多地进行数学推理
- Ped. Think Reward 实现最高的可见数学内容占比（29.76%）和提问率（9.66%）

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| BigMath | 10,000 训练 / 500 评估 | 大规模数学问题集（算术、代数、应用题） | 训练 + 核心评估 |
| WBEB (Well-Balanced Educational Benchmark) | 4 个子维度 | 教育领域综合基准（学科知识、教学知识、作文评分、教学决策） | 分布外泛化评估 |

### 实现细节

- **Backbone**: DeepSeek-R1-0528-Qwen3-8B（思考条件）/ Qwen2.5-7B（NoThink）
- **训练算法**: GRPO（veRL 框架），无单独 Value Model
- **Batch Size**: 16 problems x 8 rollouts = 128 rollouts/batch
- **最大对话轮数**: 16 turns
- **硬件**: 4 GPUs
- **Student Simulator**: Llama-3.1-8B-Instruct（以数学学生角色 prompt）
- **问题过滤**: 仅使用 Student Simulator 解题率为 1%–60% 的问题（排除过易/过难）
- **Judge**: GPT-4o-mini（Leak Judge, Help Judge, Think Judge）

### 关键发现

1. **思考过程对教学质量至关重要**: 启用思考直接从 0.120 → 0.281（+134% Delta Solve），证明推理 LLM 的内部思考对教学效果有决定性影响
2. **思考奖励与教学提示的协同效应**: 思考奖励仅在结合教学提示时才发挥最大作用（+6.9% vs 仅 +1.1%），暗示有效的思考奖励需依赖结构化的思维引导
3. **教学提示单独使用的风险**: 不使用思考奖励时，教学提示反而增加 Leak Rate（0.180 → 0.214），表明需要奖励信号来约束行为
4. **分布外泛化**: 对话式 RL 训练带来的教学推理能力可迁移到未见过的教育任务，且基本保留事实知识

---

## 批判性思考

### 优点
1. **新颖的研究视角**: 首次将 RL 优化目标从"输出质量"扩展到"思维质量"，开辟了推理 LLM 对齐的新维度
2. **教育理论的深度结合**: 将 Polya 框架融入 prompt 设计而非简单堆砌，体现了教育学与 AI 的深度融合
3. **严谨的实验设计**: 五条件消融设计清晰分离了各组件贡献，WBEB 泛化评估展示了方法的实际价值
4. **实践可行性**: 使用 GRPO（无需 Critic Model）和 4 GPU 即可训练，具有良好的可复现性

### 局限性
1. **单一学科验证**: 仅在数学辅导上验证，是否适用于其他学科（物理、编程、语言学习）未知
2. **LLM Judge 偏差**: 思考质量评估依赖 GPT-4o-mini 判断，可能存在评估偏差，缺乏人类教师的金标准对比
3. **Student Simulator 局限**: Llama-3.1-8B-Instruct 模拟学生可能无法真实反映人类学生的学习行为
4. **相对简单的教学框架**: Polya 四步法是通用框架，未针对特定年级或学习障碍进行个性化适配
5. **后续工作揭示的对齐税**: 后续研究（TEI, 2605.30666）发现相同方案应用于 DeepSeek-R1-8B 时出现严重性能退化（Delta Solve -93%）

### 潜在改进方向
1. 将 [[Thinking Reward]] 扩展到多学科教学场景，验证框架的通用性
2. 引入 [[Persona-Aware Thinking Reward]]（如后续 Special-R1 工作），针对不同学习者特征个性化思考评估
3. 用人类教师标注替代或补充 LLM Judge，提升思考质量评估的可靠性和教育效度
4. 探索更丰富的教育理论作为思考框架（如 [[Socratic Method|苏格拉底式提问]]、[[Scaffolding Theory|支架式教学]]）

### 可复现性评估
- [x] 代码开源（OpenLearnLM 仓库）
- [x] 训练细节完整（附录含超参数）
- [x] 数据集可获取（BigMath, WBEB）
- [ ] 预训练模型（未见发布）

---

## 关联笔记

### 基于
- [[PedagogicalRL]]: 首个将 RL 应用于 LLM 辅导训练的工作，奠定了 Leak + Help 二元教学评估范式
- [[DeepSeek-R1]]: 推理 LLM 的基础模型，其 `<think>` 推理机制是本文的核心操作对象
- [[GRPO]]: 核心训练算法，组内相对策略优化，无需独立价值模型
- [[veRL]]: 训练框架，高效实现 GRPO 的对话级 rollout

### 对比
- [[Qwen2.5-7B-Instruct]]: NoThink 基线，代表标准指令微调模型的辅导能力

### 方法相关
- [[Thinking Reward]]: 核心创新，将 RL 奖励扩展至推理轨迹
- [[Polya's Problem-Solving Framework]]: 教育理论支撑，结构化思维引导
- [[LLM Judge]]: 自动化评估基础设施（Leak Judge, Help Judge, Think Judge）
- [[Student Simulator]]: 低成本批量评估学生辅导效果

### 硬件/数据相关
- [[BigMath]]: 大规模数学训练数据集
- [[WBEB]]: 教育领域综合评估基准

---

## 速查卡片

> [!summary] PedagogicalRL-Thinking
> - **核心**: 首次用 RL 优化 LLM 的"思考过程"而非仅"输出答案"，结合 Polya 教育框架，训练更好的 AI 数学导师
> - **方法**: GRPO + 三部分复合奖励（解题效果 + 教学规范 + 思考质量）+ 教育理论引导的思维提示
> - **结果**: Delta Solve 0.294（最高），Leak 0.172（最低），Helpful 0.776（最高）；仅在数学对话上训练却在教育基准上展现出显著的分布外泛化
> - **代码**: [OpenLearnLM](https://anonymous.4open.science/repo/pedagogical_reasoning_submission-2787/)

---

*笔记创建时间: 2026-08-01*
