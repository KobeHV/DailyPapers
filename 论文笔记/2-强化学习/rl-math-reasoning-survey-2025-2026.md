# RL 强化学习提升数学解题与教育解题能力 —— 近一年论文调研笔记

> 调研时间范围：2025 年 1 月 – 2026 年 7 月
> 整理日期：2026-07-15

---

## 目录

1. [标志性 / 里程碑论文](#1-标志性--里程碑论文)
2. [RL 算法与训练方法](#2-rl-算法与训练方法)
3. [奖励设计：从结果到过程](#3-奖励设计从结果到过程)
4. [小模型的 RL 数学推理](#4-小模型的-rl-数学推理)
5. [教育应用：智能辅导与自适应教学](#5-教育应用智能辅导与自适应教学)
6. [综述与基准](#6-综述与基准)
7. [关键争议与反思](#7-关键争议与反思)
8. [核心趋势总结](#8-核心趋势总结)

---

## 1. 标志性 / 里程碑论文

### 1.1 DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **发表**: arXiv 2025.01 → **Nature 封面** (Vol. 645, 2025.09)
- **作者**: DeepSeek-AI (通讯: 梁文锋)
- **核心贡献**: 首次证明**纯 RL（无 SFT 监督微调）**可以激发 LLM 的推理能力
- **方法**:
  - **DeepSeek-R1-Zero**: 在 DeepSeek-V3 Base 上直接用 GRPO + 规则准确率奖励训练（数学题答案对/错），完全不使用人类标注的推理示范
  - **DeepSeek-R1**: 多阶段流水线——冷启动 SFT 数据 → 推理导向 RL → 拒绝采样 + SFT → 最终对齐 RL
- **关键结果**:
  - R1-Zero 在 AIME 2024 上 pass@1 从 15.6% → **77.9%**（超过人类平均选手）
  - 观察到"**Aha Moment**"：训练过程中模型自发学会说"wait"来暂停思考、重新审视
- **影响力**: 首篇通过 Nature 同行评审的 LLM 论文，引爆了整个 RL-for-reasoning 方向

### 1.2 rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking

- **发表**: ICML 2025 **Oral** (Microsoft)
- **核心贡献**: 证明小模型（7B）不依赖大模型蒸馏也能达到 o1 级别数学推理
- **方法**:
  - **MCTS** 测试时搜索 + **Process Preference Model (PPM)** 引导
  - 代码增强的 CoT 数据合成（大规模 MCTS rollout）
  - **自进化**: 策略 SLM 和 PPM 从零开始，4 轮迭代相互提升
- **关键结果** (Qwen2.5-Math-7B):
  - MATH: 58.8% → **90.0%**（超越 o1-preview +4.5%）
  - AIME: 解出 **53.3% (8/15)**，进入高中数学竞赛者前 20%
- **代码**: [github.com/microsoft/rStar](https://github.com/microsoft/rStar)

### 1.3 Open-Reasoner-Zero

- **发表**: NeurIPS 2025, arXiv 2503.24290
- **核心贡献**: 首个**完全开源的大规模 RL-from-base-model** 推理训练方案
- **方法**:
  - 极简设计：Vanilla PPO + GAE (λ=1, γ=1)，**无 KL 正则、无 SFT、无格式奖励**
  - 仅用二元奖励（答案对=1，错=0），不做奖励工程
- **关键结果** (32B):
  - AIME 2024: **48.1**（vs DeepSeek-R1-Zero-Qwen-32B 的 47.0）
  - 训练效率约为 DeepSeek-R1-Zero 的 **1/10**
  - 0.5B 到 32B 全系列均有效，跨 Qwen/Llama/Mistral/DeepSeek-Math
- **代码**: [github.com/openreasoner/openr](https://github.com/openreasoner/openr)

### 1.4 rStar2-Agent (Microsoft)

- **发表**: 2025.07
- **核心贡献**: 14B 模型通过 agentic RL 达到 DeepSeek-R1 671B 水平
- **方法**: GRPO-RoC (Resample-on-Correct) 新 RL 算法，仅 **510 RL 步**、一周 64×MI300X GPU 训练
- **关键结果**: AIME24: **80.6%** | AIME25: **69.8%**，均超越 DeepSeek-R1 671B

---

## 2. RL 算法与训练方法

### 2.1 DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

- **发表**: arXiv 2503.14476 (2025.03)
- **团队**: Qiying Yu, Zheng Zhang 等 35 位作者（字节跳动等）
- **核心贡献**: 开源可复现的大规模 LLM RL 训练系统，基于 `verl` 框架
- **四项关键技术**: Decoupled Clip、Dynamic Sampling、Token-level Policy Gradient Loss、Overlong Reward Shaping
- **关键结果**: Qwen2.5-32B → AIME 2024 **50 分**
- **代码**: 完全开源

### 2.2 DAPO (Direct Advantage-Based Policy Optimization)

- **发表**: NeurIPS 2025, arXiv 2412.18279
- **作者**: Jiacai Liu 等 (Skywork AI / 复旦大学)
- **核心贡献**: **步骤级离线 RL**——用 Critic 预测每一步的推理准确率，提供密集步骤级信号
- **区别于前一个 DAPO**: 这是步骤级方法，解决稀疏奖励问题；Actor 和 Critic 独立训练避免共训不稳定

### 2.3 GRPO-LEAD: Difficulty-Aware RL for Concise Mathematical Reasoning

- **发表**: arXiv 2504.09696 (2025)
- **核心贡献**: 解决 GRPO 的奖励稀疏和冗长问题
- **三项改进**:
  1. 长度依赖的准确率奖励（鼓励简洁推理）
  2. 错误答案显式惩罚
  3. 难度感知的 advantage 重加权

### 2.4 DASH: Effective RL for Reasoning

- **发表**: arXiv 2505.17218 (2025.05)
- **核心贡献**: Preemptive Sampling + Gradient Filtering
- **关键结果**: 相比标准 GRPO **减少 83% 训练时间**，准确率无损

### 2.5 Self-Evolving Curriculum for LLM Reasoning

- **发表**: arXiv 2505.14970 (2025)
- **核心贡献**: 将 RL 微调中的课程选择建模为**非稳态多臂老虎机**问题
- 自动学习在规划、归纳推理、数学等领域中何时呈现何种难度的题目

### 2.6 算法对比研究

- **Comparative Analysis of PPO, GRPO, and DAPO** (arXiv 2512.07611, 2025.12):
  - 系统对比三种算法在 LLM 推理上的表现
  - 关键发现：DAPO 中的 Dynamic Sampling 组件**并未提升性能**，禁用后反而更好

---

## 3. 奖励设计：从结果到过程

### 3.1 奖励粒度：过程 vs 结果

**Reward Granularity in RLVR** (arXiv 2607.02869, 2026.07):
- 在 Qwen2.5-0.5B + GRPO + GSM8K 上对比
- **过程级监督 63.73% vs 结果级 53.75%**（~10 个百分点差距）
- 过程奖励带来更高的步骤有效性和推理追踪保真度

### 3.2 Negative Reinforcement（负强化）

**The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning** (NeurIPS 2025, arXiv 2506.01347):
- 作者: Xinyu Zhu, Mengzhou Xia, Danqi Chen 等 (Princeton)
- **惊人发现**: **仅用错误样本训练**（惩罚错答，不奖励对答）可以匹配甚至超越 PPO/GRPO
- 惩罚错误答案 → Pass@k 多样性提升
- 奖励正确答案 → 高 k 时多样性退化
- **启示**: 抑制错误路径、重新分配概率质量本身就是强大的学习机制

### 3.3 独特性感知奖励

**Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving** (ACL 2026 Findings):
- 奖励**稀有/新颖的解题策略**而非仅正确性
- 按策略聚类 rollout，按簇大小反向加权奖励
- 防止探索坍缩，提升 pass@k 和 AUC（数学、物理、医学推理）

### 3.4 质量感知奖励

**Forge: Quality-Aware RL for NP-Hard Optimization** (ACL 2026 Findings):
- 超越二元正确性，使用质量感知奖励
- 相比二元奖励提升 **28.8%**，训练迁移到数学 (+2.2%)

### 3.5 Spurious Rewards（伪奖励争议）⚠️

**Spurious Rewards: Rethinking Training Signals in RLVR** (ICML 2026, arXiv 2506.10947):
- 作者: UW / Allen AI / Berkeley 联合团队
- **震撼发现**: 使用**随机抛硬币**或**故意错误的奖励**训练 Qwen2.5-Math-7B，MATH-500 仍提升 **21-24%**（真奖励 +29%）
- **原因分析**:
  1. GRPO clipping bias → 放大预训练中已有的高先验行为
  2. Qwen-Math 特有的 "**code reasoning**" 能力（生成 Python 代码但不执行）
- **关键局限**: 仅在 Qwen 上有效，Llama/OLMo 上无效甚至有害
- **后续**: 有论文认为这暴露了预训练数据污染导致的记忆捷径（"Anchor-Adapter Circuit"）
- **影响**: 迫使社区重新审视大量仅基于 Qwen 验证的 RLVR 论文

---

## 4. 小模型的 RL 数学推理

### 4.1 1-Shot RLVR（单样本 RL）

**Reinforcement Learning for Reasoning with One Training Example** (NeurIPS 2025 Most Influential):
- **仅用 1 个训练样本** + RLVR → Qwen2.5-Math-1.5B 在 MATH500 上从 36.0% → **73.6%**
- 2 个样本 → 74.8%，匹配使用 1200 样本训练的效果
- 发现"**后饱和泛化**"现象：训练准确率饱和后性能继续提升

### 4.2 Teaching Small LLMs to Reason Using RL

- **发表**: IEEE 2026.01 (Las Vegas)
- Qwen2.5-0.5B: 0.08% → **27.64%** GSM8K（超越 GPT-3 175B）
- Qwen2.5-3B: 55.1% → **74.6%**（超越 Minerva 540B、LLaMA-2-70B）

### 4.3 Smaller Models are Natural Explorers

- **发表**: arXiv 2605.30789 (2026.05)
- 提出 **S2L-PO (Small-to-Large Policy Optimization)**
- 用固定的小模型（1.7B）作为"天然探索者"生成多样化 rollout 来训练大模型
- AIME24 上 +8.8%（用 1.7B explorer 引导 8B 模型），同时减少 rollout 计算量

### 4.4 Weak-to-Strong Elicitation via Mismatched Wrong Drafts

- **发表**: arXiv 2605.17314 (2026.05)
- 将小模型（Qwen2.5-Math-1.5B）的**错解草稿**注入大模型（Mathstral-7B）的 GRPO 上下文
- MATH-500: **71.98%**（Mathstral-7B 最佳公开结果，超越 WizardMath）
- 不匹配的错误草稿独特地提升了 OOD AIME 2025/2026 的 pass@k

---

## 5. 教育应用：智能辅导与自适应教学

这是 2025-2026 年增长最快的方向之一，RL 开始从"让模型会解题"走向"让模型会教题"。

### 5.1 UCO: Multi-Turn Interactive RL for Adaptive Teaching

- **发表**: arXiv 2511.08873 (2025.11 / 2026.01)
- **核心贡献**: **Unidirectional Cognitive Optimization**，多轮交互式 RL 教学框架
- **两个新奖励函数**:
  - **Progress Reward**: 捕捉学生是否真正从困惑到理解
  - **Scaffold Reward**: 动态识别每个学生的最近发展区 (ZPD)
- **评估**: BigMath + MathTutorBench，超越所有同规模模型，匹敌闭源大模型
- **教育理论结合**: 维果茨基 ZPD、支架式教学 (Scaffolding)

### 5.2 PedagogicalRL-Thinking: 奖励模型的"教学思维"

- **发表**: arXiv 2601.14560 (2026.01)
- **作者**: Unggi Lee 等
- **两项创新**:
  - **Pedagogical Reasoning Prompting**: 用教育理论引导推理过程（非通用指令）
  - **Thinking Reward**: 显式奖励推理轨迹的**教学质量**
- **关键发现**: 仅在数学辅导对话上训练的模型**迁移**到未见过的教育基准，同时保持事实知识

### 5.3 RL-Based Alignment for LLM Tutors (MathTutorBench)

- **作者**: Jakub Macina 等 (ETH Zurich)
- **方法**: 在线 RL + 模拟学生-导师交互，**无需人类标注**
- 7B 导师模型性能可比肩大得多的闭源模型（如 LearnLM）
- 引入**可控奖励加权**来平衡教学支持 vs 学生自主解题（Pareto 前沿）

### 5.4 真实部署: AI Tutors via LLM-Guided RL

- **作者**: Chung, Zhang, Kung, Bastani & Bastani (Stanford)
- **部署**: 与**台北市政府**合作，10 所高中 Python 教学
- **方法**: GenAI 聊天机器人 + RL 算法为练习排序，信号来自学生-聊天机器人交互
- **RCT 结果**: 自适应排序使独立期末考试提升 **+0.15 SD**（≈6-9 个月学龄）

### 5.5 Tutoring Effectiveness Index (TEI)

- **发表**: arXiv 2605.30666 (2026.05)
- **核心贡献**: **无需训练、无需 judge 模型**的四信号指数评估 LLM 数学导师质量
- **四个信号**: Schoenfeld-Verify 关键词比、数学步骤密度、反问率、深度推理门
- **效果**: N=8 时，预错误场景改进率从 59% → **81.9%**（使用冻结的 DeepSeek-R1-8B）

### 5.6 History-Aware Student Simulation

- **发表**: arXiv 2605.30051 (2026.05)
- **作者**: UMass Amherst + Eedi (真实在线数学平台数据)
- **方法**: 用 RL (GRPO) 训练 **Profile Generator** + **Student Simulator**
- Profile 捕获知识状态、错误概念、对话行为、语言风格
- 用于以极低成本替代真实学生进行导师训练/评估

### 5.7 Hybrid DRL-GenAI for Adaptive Tutoring in VR

- **发表**: iLRN 2026
- **方法**: **语义解耦**——Deep RL 处理策略性教学决策 + 生成式 AI 处理情境化支架
- **评估**: VR 中向量空间/线性代数教学，156 次干预，9 名用户
- **结果**: 教学连贯性 Cohen's Kappa = **1.00**，学生满意度 **4.66/5.00**

### 5.8 Step-Aware ITSs with Synthetic Step-by-Step Solutions

- **发表**: ACM Learning @ Scale 2025.07
- **方法**: 用 GPT-o3-mini 生成合成逐步解答（Junyi Academy 数据集）
- **结果**: 模拟学生（Llama 不同知识水平）在逐步引导下多解决 **42%** 的练习

---

## 6. 综述与基准

### 6.1 重要综述

| 论文 | 时间 | 核心覆盖 |
|------|------|---------|
| **Towards Large Reasoning Models: A Survey of Reinforced Reasoning with LLMs** (arXiv 2501.09686) | 2025.01 | 20 位作者，覆盖自动数据构建、学习推理技术、测试时扩展、开源项目 |
| **A Survey of RL for Large Reasoning Models** (arXiv 2509.08827) | 2025 | 40+ 位作者，覆盖 RL 核心问题（奖励黑客、探索）、训练资源、下游应用 |
| **The Periodic Table of LLM Reasoning** (arXiv 2606.11470) | 2026.06 | 300+ 篇论文系统分类，涵盖 CoT、数学、代码、工具、智能体、RL 推理及失败模式 |
| **A Survey of Process Reward Models** (ACL 2026, arXiv 2510.08049) | 2025-2026 | 全闭环框架：过程数据生成→PRM 构建→PRM 使用（测试时扩展+RL） |
| **Enhancing LLM Reasoning with Reward Models: An Analytical Survey** (arXiv 2510.01925) | 2025 | 提出 RARL (Reasoning-Aligned RL) 框架 |
| **Generate, Filter, Control, Replay: Rollout Strategies for LLM RL** | 2025-2026 | GFCR 生命周期分类法，优化器无关的 rollout 策略 |

### 6.2 重要基准

| 基准 | 方向 | 特点 |
|------|------|------|
| **MMTutorBench** (NAACL 2025) | AI 数学辅导 | 770 题，6 维度评估（洞察发现、操作表述、操作执行等），Rubric-based LLM-as-Judge |
| **MathTutorBench** | 辅导评估 | 开源，全面辅导质量评估 |
| **MATH-Beyond** (arXiv 2510.11653) | RL 扩展 | 测试 RL 是否超越基座模型能力边界 |
| **PRMBench** | 过程奖励 | 6K+ 题，80K 步骤标注，多维标签 |
| **ProcessBench** | 过程奖励 | 竞赛级数学，最早错误检测 |
| **Socratic-PRMBench** | 过程奖励 | 3K 缺陷轨迹，6 种错误模式 |

---

## 7. 关键争议与反思

### 7.1 RLVR 到底学到了什么？

Spurious Rewards 论文引发了根本性反思: RLVR 的性能提升到底是**学会推理**还是**激发预训练记忆中已有的能力**？GRPO 的 clipping 机制可能只是一个保守重加权器，不能发现真正新的推理模式。

**后续影响**:
- 许多 RLVR 论文（TTRL, 1-shot RL 等）仅基于 Qwen 验证，其结论可能不具普适性
- 社区被呼吁跨多种模型家族验证

### 7.2 数学推理能否泛化？

**Does Learning Mathematical Problem-Solving Generalize to Broader Reasoning?** (arXiv 2507.04391, 2025.07):
- **长 CoT + 规则 RL 数学训练**显著提升跨领域通用推理
- **短 CoT 指令微调**反而可能损害泛化能力
- 结论: RL 训练的推理模式比 SFT 更具迁移性

### 7.3 概念理解 vs 模式匹配

**CORE: Concept-Oriented Reinforcement** (arXiv 2512.18857, 2025.12):
- LLM 能背诵定义但无法正确应用概念（"定义-应用鸿沟"）
- CORE 在 RL rollout 中注入概念片段 + 概念对齐测验
- 在三类模型 (Qwen, DeepSeek, Llama) 上均有一致提升
- **教育启示**: 当前 RLVR 强化的更可能是模式匹配而非真正的概念理解

---

## 8. 核心趋势总结

### 8.1 技术趋势

```
RLVR 主导 → 奖励设计精细化 → 教育应用兴起 → 反思与理论深化
 (2025H1)      (2025H2)         (2026H1)        (2026)
```

| 趋势 | 代表论文 | 成熟度 |
|------|---------|--------|
| **RLVR (RL with Verifiable Rewards)** | DeepSeek-R1, Open-Reasoner-Zero | 🟢 成熟 |
| **GRPO 成为主流算法** | 多数 2025 论文，从 0.5B 到 671B 均验证 | 🟢 成熟 |
| **过程奖励 > 结果奖励** | Reward Granularity, PRM Survey, DAPO(2) | 🟡 快速发展 |
| **数据效率极致化** | 1-Shot RLVR, Negative RL | 🟡 快速发展 |
| **负样本/探索多样性** | Negative RL, Uniqueness-Aware RL, S2L-PO | 🟡 快速发展 |
| **RL 用于教学而不仅是解题** | UCO, PedagogicalRL, MathTutorBench | 🔴 新兴 |
| **PRM 作为 RL 的密集信号** | PRM Survey (全闭环) | 🟡 快速发展 |
| **从数学 RL 向其他领域迁移** | Reasoning Curriculum, Crossing the Reward Bridge | 🟡 快速发展 |
| **开源生态建设** | OpenR, verl, Open-Reasoner-Zero | 🟢 成熟 |

### 8.2 值得关注的开源项目

| 项目 | 链接 | 说明 |
|------|------|------|
| OpenR | [github.com/openreasoner/openr](https://github.com/openreasoner/openr) | 统一 RL+PRM+搜索框架 |
| rStar | [github.com/microsoft/rStar](https://github.com/microsoft/rStar) | MCTS + 自进化推理 |
| verl | GitHub - volcengine | DAPO 和多数 GRPO 训练的底层框架 |
| Rethink-RLVR | [github.com/ruixin31/Rethink_RLVR](https://github.com/ruixin31/Rethink_RLVR) | Spurious Rewards 复现与分析 |

### 8.3 教育应用的关键转变

1. **从"模型会解题"到"模型会教题"** — RL 用于优化教学策略本身
2. **从单轮答案到多轮支架** — UCO、PedagogicalRL 都是多轮交互 RL
3. **从人工评估到模拟学生** — History-Aware Profiles、RL 训练的 student simulator 大幅降低评估成本
4. **从实验室到真实课堂** — 台北市 RCT 是重要里程碑 (+0.15 SD)
5. **教育理论深度融合** — ZPD、Scaffolding、Productive Failure 等概念进入奖励函数设计

### 8.4 待解决问题

- RLVR 的增益多少来自真正的推理学习 vs 预训练能力的激发？（Spurious Rewards 问题未解决）
- 过程奖励模型 (PRM) 的标注成本仍然高，隐式 PRM 方向有前景但仍不成熟
- RL 训练的推理模型在 OOD（分布外）场景下的泛化仍有限
- 教育场景缺乏大规模、高质量的**多轮教学交互数据**
- 如何将"教学思维"（pedagogical reasoning）编码进奖励函数仍需探索

---

## 参考文献（按首次出现顺序）

1. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL." Nature 645, 2025.
2. Guan et al. "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking." ICML 2025 (Oral).
3. Hu, Zhang et al. "Open-Reasoner-Zero: An Open Source Approach to Scaling Up RL on the Base Model." NeurIPS 2025, arXiv:2503.24290.
4. Yu et al. "DAPO: An Open-Source LLM RL System at Scale." arXiv:2503.14476, 2025.
5. Liu et al. "DAPO: Improving Multi-Step Reasoning with Direct Advantage-Based Policy Optimization." NeurIPS 2025, arXiv:2412.18279.
6. "GRPO-LEAD: A Difficulty-Aware RL Approach for Concise Mathematical Reasoning." arXiv:2504.09696, 2025.
7. "DASH: Effective RL for Reasoning in Language Models." arXiv:2505.17218, 2025.
8. "Self-Evolving Curriculum for LLM Reasoning." arXiv:2505.14970, 2025.
9. "Reward Granularity in RLVR: Process vs Outcome Reward Structures." arXiv:2607.02869, 2026.
10. Zhu, Xia, Chen et al. "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning." NeurIPS 2025, arXiv:2506.01347.
11. "Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving." ACL 2026 Findings.
12. "Forge: Quality-Aware RL for NP-Hard Optimization in LLMs." ACL 2026 Findings.
13. Shao et al. "Spurious Rewards: Rethinking Training Signals in RLVR." ICML 2026, arXiv:2506.10947.
14. Wang et al. "RL for Reasoning in LLMs with One Training Example." NeurIPS 2025.
15. "Teaching Small LLMs to Reason Using RL." IEEE 2026.
16. "Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO." arXiv:2605.30789, 2026.
17. "Weak-to-Strong Elicitation via Mismatched Wrong Drafts." arXiv:2605.17314, 2026.
18. Wei et al. "UCO: Multi-Turn Interactive RL for Adaptive Teaching with LLMs." arXiv:2511.08873, 2025.
19. Lee et al. "Rewarding How Models Think Pedagogically." arXiv:2601.14560, 2026.
20. Macina et al. "RL-Based Alignment Framework for LLM Tutors." 2025.
21. Chung et al. "Effective Personalized AI Tutors via LLM-Guided RL." 2025.
22. Jaechang & Lee. "The Tutoring Effectiveness Index." arXiv:2605.30666, 2026.
23. Duan et al. "History-Aware Profiles for Student Simulation." arXiv:2605.30051, 2026.
24. "Step-Aware ITSs: Synthetic Step-by-Step Exercise Solutions." ACM Learning @ Scale 2025.
25. Zheng et al. "A Survey of Process Reward Models." ACL 2026, arXiv:2510.08049.
26. Xu et al. "Towards Large Reasoning Models: A Survey." arXiv:2501.09686, 2025.
27. Zhang et al. "A Survey of RL for Large Reasoning Models." arXiv:2509.08827, 2025.
28. Anand et al. "The Periodic Table of LLM Reasoning." arXiv:2606.11470, 2026.
29. Liu et al. "Enhancing LLM Reasoning with Reward Models: An Analytical Survey." arXiv:2510.01925, 2025.
30. Zhou et al. "Does Learning Mathematical Problem-Solving Generalize?" arXiv:2507.04391, 2025.
31. Gao et al. "CORE: Concept-Oriented Reinforcement." arXiv:2512.18857, 2025.
32. "QuestA: Question Augmentation for RL Reasoning." ICLR 2026.
33. "SATURN: SAT-based RL to Unleash LLMs Reasoning." NeurIPS 2025.
34. "SwS: Self-aware Weakness-driven Problem Synthesis in RL." NeurIPS 2025.
35. "Crossing the Reward Bridge: Expanding RLVR Across Diverse Domains." arXiv:2503.23829, 2025.
36. Macina et al. "MMTutorBench." NAACL 2025.
37. "Teaching LLM to Reason: RL from Algorithmic Problems without Code (TeaR)." arXiv:2507.07498, 2025.
38. "Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math." arXiv:2510.26143, 2025.
