---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
method_name: "DeepSeekMath"
authors: [Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo]
year: 2024
venue: arXiv
tags: [mathematical-reasoning, reinforcement-learning, grpo, language-model, pre-training, data-curation, chain-of-thought]
zotero_collection: 3-大模型架构
image_source: local
arxiv_html: https://arxiv.org/abs/2402.03300
created: 2026-07-10
---

# 论文笔记：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | DeepSeek-AI, Tsinghua University, Peking University |
| 日期 | February 2024 |
| 项目主页 | https://github.com/deepseek-ai/DeepSeek-Math |
| 对比基线 | [[Minerva]], [[Llemma]], [[WizardMath]], [[MetaMath]] |
| 链接 | [arXiv](https://arxiv.org/abs/2402.03300) / [Code](https://github.com/deepseek-ai/DeepSeek-Math) |

---

## 一句话总结

> 通过从 Common Crawl 中大规模挖掘高质量数学预训练数据（120B tokens）并引入 GRPO 强化学习算法，DeepSeekMath 7B 在 MATH 上达到 51.7%，逼近 GPT-4 水平。

---

## 核心贡献

1. **DeepSeekMath Corpus 构建**: 提出了一个迭代式数据筛选 pipeline，从 Common Crawl 中提取 120B tokens 的高质量数学语料，规模是 Minerva 的 7 倍、OpenWebMath 的 9 倍。
2. **[[GRPO|Group Relative Policy Optimization (GRPO)]]**: 提出一种无需 critic 模型的 PPO 变体，通过组内奖励归一化估计 baseline，大幅降低训练资源消耗。
3. **统一范式分析**: 为 SFT、RFT、DPO、PPO、GRPO 等不同方法提供了统一的数学框架，并深入研究了在线 vs 离线训练、outcome vs process supervision、单轮 vs 迭代 RL 等关键因素。

---

## 问题背景

### 要解决的问题
如何让开源语言模型在数学推理任务上达到接近 GPT-4 和 Gemini-Ultra 的水平。

### 现有方法的局限
- 闭源模型（GPT-4, Gemini-Ultra）性能优异但不可公开获取
- 开源模型（Mistral 7B, Llemma 34B）在 MATH 等竞赛级基准上表现远落后于闭源模型
- 已有的数学预训练语料（如 MathPile、OpenWebMath、Proof-Pile-2）规模有限，且多为英文单语
- PPO 中的 value function (critic) 模型与 policy 规模相当，带来巨大的内存和计算开销

### 本文的动机
- 公共 Common Crawl 数据中蕴含大量未被充分利用的高质量数学内容
- 通过精心设计的数据筛选 pipeline 可以大幅提升小模型的数学推理能力
- 去掉 value function 的 RL 算法能在保持性能的同时显著降低资源需求

---

## 方法详解

### 数据收集与模型训练总览

DeepSeekMath 采用 **三阶段训练** 范式：

1. **Math Pre-Training**: 以 DeepSeek-Coder-Base-v1.5 7B 为初始化，在 500B tokens（56% 数学 + 4% AlgebraicStack + 10% arXiv + 20% GitHub 代码 + 10% 自然语言）上继续预训练
2. **Supervised Fine-Tuning (SFT)**: 在 776K 数学指令数据上微调，数据覆盖 CoT、PoT、Tool-Integrated 三种格式
3. **Reinforcement Learning (GRPO)**: 在 144K GSM8K/MATH 问题上进行 RL 训练

- **总参数**: 7B
- **Backbone**: DeepSeek-Coder-Base-v1.5 7B (基于 [[DeepSeek LLM]] 架构，使用 [[Multi-Head Attention]], [[RoPE]], [[SwiGLU]], [[GQA]])

### 核心模块 1: 迭代式数学数据收集 Pipeline

**设计动机**: 从 Common Crawl 中系统性挖掘高质量数学网页，通过迭代提升分类器质量。

**具体实现**:
- 以 [[OpenWebMath]] 作为种子语料，训练 [[fastText]] 二分类器
- 对 Common Crawl 进行 URL 去重和近似去重，保留 40B HTML 页面
- 用 fastText 对 CC 页面打分，保留 top-ranking 页面
- 识别 math-related domains（收集率 >10% 的域名），人工标注该域内数学相关 URL，补充到种子语料
- 经 4 轮迭代后得到 35.5M 数学网页（120B tokens），第 4 轮 98% 数据已在第 3 轮中收集

### 核心模块 2: [[GRPO|Group Relative Policy Optimization (GRPO)]]

**设计动机**: 消除 PPO 中 critic model 的依赖，通过组内相对奖励估计 baseline，节省训练资源。

**具体实现**:
- 对每个问题 $q$，从旧策略 $\pi_{\theta_{old}}$ 采样 $G$ 个输出 $\{o_1, o_2, \dots, o_G\}$
- 用 reward model 给每个输出打分，得到 $G$ 个奖励 $\mathbf{r} = \{r_1, r_2, \dots, r_G\}$
- 对组内奖励进行归一化：$\widetilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$
- Outcome Supervision: 将所有 token 的 advantage 设为 $\hat{A}_{i,t} = \widetilde{r}_i$
- Process Supervision: 每个 step 结束时给出奖励，$\hat{A}_{i,t} = \sum_{index(j) \geq t} \widetilde{r}_i^{index(j)}$
- 在 loss 中直接添加 KL 散度正则化项（而非像 PPO 那样加到 reward 中），使用无偏 KL 估计器

---

## 关键公式

### 公式 1: [[PPO|PPO 代理目标函数]]

$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}\left[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)\right] \frac{1}{|o|} \sum_{t=1}^{|o|} \min\left[\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})} A_t, \text{clip}\left(\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}, 1-\varepsilon, 1+\varepsilon\right) A_t\right]
$$

**含义**: PPO 的标准代理目标函数，通过重要性采样和 clipping 约束策略更新幅度。

**符号说明**:
- $\pi_\theta, \pi_{\theta_{old}}$: 当前策略和旧策略
- $q, o$: 问题和模型输出
- $A_t$: 基于 GAE 计算的 advantage
- $\varepsilon$: clipping 超参数

### 公式 2: [[GRPO|GRPO 代理目标函数]]

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Bigg[& q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q) \Bigg] \\
\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Bigg\{&\min\left[\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t}\right] \\
&- \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] \Bigg\}
\end{aligned}
$$

**含义**: GRPO 从组内采样的多个输出中估计 advantage，并用 KL 散度正则化，无需 critic 模型。

**符号说明**:
- $G$: 每组采样的输出数量
- $\hat{A}_{i,t}$: 基于组内相对奖励计算的 advantage
- $\beta$: KL 惩罚系数
- $\pi_{ref}$: 参考模型（通常为初始 SFT 模型）

### 公式 3: [[KL 散度|无偏 KL 估计器]]

$$
\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1
$$

**含义**: GRPO 使用的无偏 KL 散度估计，保证恒为正，可以直接加到 loss 中而非混入 reward 计算。

### 公式 4: [[统一训练范式|统一训练梯度公式]]

$$
\nabla_\theta \mathcal{J}_\mathcal{A}(\theta) = \mathbb{E}_{(q,o) \sim \mathcal{D}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} GC_\mathcal{A}(q,o,t,\pi_{rf}) \nabla_\theta \log \pi_\theta(o_t|q,o_{<t}) \right]
$$

**含义**: 几乎所有训练方法（SFT、RFT、DPO、PPO、GRPO）都可以统一为"数据来源 × 梯度系数 × 对数概率梯度"的形式。

**符号说明**:
- $\mathcal{D}$: 数据来源（离线 SFT 数据 / 在线策略采样）
- $GC_\mathcal{A}$: 梯度系数，由算法和奖励函数决定
- $\pi_{rf}$: 奖励函数（rule-based 或 model-based）

---

## 关键图表

### Figure 1: Top1 Accuracy on MATH (开源 vs 闭源模型)

![[DeepSeekMath_fig1.png]]

**说明**: 展示了 DeepSeekMath 7B 在 MATH 基准上以 51.7% 的准确率超越所有开源模型，逼近 GPT-4 (52.9%) 和 Gemini-Ultra (53.2%) 的水平。

### Figure 2: 迭代式数据收集 Pipeline

![[DeepSeekMath_fig2_pipeline.png]]

**说明**: 展示从 Common Crawl 中系统收集数学网页的迭代式 pipeline。以 OpenWebMath 为种子训练 fastText 分类器，挖掘更多数学页面，通过人工标注改进种子质量，共 4 轮迭代。

### Figure 3: Benchmark Curves of DeepSeek-LLM 1.3B

![[DeepSeekMath_fig3_curves.png]]

**说明**: 不同数学语料库上训练的 1.3B 模型在 GSM8K、MATH、CMATH、MMLU-STEM 等基准上的表现曲线。DeepSeekMath Corpus 展现出更陡峭的学习曲线和更持久的改进。

### Figure 4: PPO vs GRPO 架构对比

![[DeepSeekMath_fig4_ppo_grpo.png]]

**说明**: PPO 需要同时维护 policy model、value function (critic) 和 reward model，而 GRPO 通过组内输出奖励的归一化直接估计 baseline，完全去掉了 value function。

### Figure 5: 不同方法性能对比

![[DeepSeekMath_fig5_methods_comparison.png]]

**说明**: DeepSeekMath-Instruct 1.3B 在使用不同方法（RFT、DPO、Online RFT、GRPO + OS、GRPO + PS）继续训练后的 GSM8K 和 MATH 性能。GRPO + Process Supervision 表现最佳。

### Figure 6: 迭代强化学习性能

![[DeepSeekMath_fig6_iterative_rl.png]]

**说明**: 迭代 GRPO 在 GSM8K 和 MATH 上的性能提升，第一轮迭代带来了最显著的改进。

### Figure 7: Maj@K 和 Pass@K 分析

![[DeepSeekMath_fig7_majk_passk.png]]

**说明**: SFT 和 RL 模型在 GSM8K 和 MATH 上的 Maj@K 和 Pass@K 对比。RL 提升了 Maj@K 但未提升 Pass@K，说明 RL 增强了输出分布的鲁棒性而非底层能力。

### Table 1: 数学语料库质量对比

| Math Corpus | Size | GSM8K | MATH | OCW | SAT | MMLU STEM | CMATH | Gaokao MathCloze | Gaokao MathQA |
|-------------|------|-------|------|-----|-----|-----------|-------|-----------------|---------------|
| No Math Training | N/A | 2.9% | 3.0% | 2.9% | 15.6% | 19.5% | 12.3% | 0.8% | 17.9% |
| MathPile | 8.9B | 2.7% | 3.3% | 2.2% | 12.5% | 15.7% | 1.2% | 0.0% | 2.8% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 3.7% | 31.3% | 29.6% | 16.8% | 0.0% | 14.2% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 3.7% | 43.8% | 29.2% | 19.9% | 5.1% | 11.7% |
| **DeepSeekMath Corpus** | **120.2B** | **23.8%** | **13.6%** | **4.8%** | **56.3%** | **33.1%** | **41.5%** | **5.9%** | **23.6%** |

**说明**: DeepSeekMath Corpus 在所有基准上全面领先，且具有多语言能力（中英文均有显著提升）。

### Table 2: Base Model 数学推理对比

| Model | Size | GSM8K | MATH | OCW | SAT | MMLU STEM | CMATH | Gaokao MathCloze | Gaokao MathQA |
|-------|------|-------|------|-----|-----|-----------|-------|-----------------|---------------|
| Minerva 540B | 540B | 58.8% | 33.6% | 17.6% | - | 63.9% | - | - | - |
| Mistral 7B | 7B | 40.3% | 14.3% | 9.2% | 71.9% | 51.1% | 44.9% | 5.1% | 23.4% |
| Llemma 34B | 34B | 54.0% | 25.3% | 10.3% | 71.9% | 52.9% | 56.1% | 11.9% | 26.2% |
| **DeepSeekMath-Base 7B** | **7B** | **64.2%** | **36.2%** | **15.4%** | **84.4%** | **56.5%** | **71.7%** | **20.3%** | **35.3%** |

**说明**: DeepSeekMath-Base 7B 在全部 8 个基准上超越所有开源 base 模型，并在 MATH 上超越 77 倍大的 Minerva 540B。

### Table 3: Tool Use 和形式化定理证明

| Model | Size | GSM8K+Python | MATH+Python | miniF2F-valid | miniF2F-test |
|-------|------|-------------|------------|---------------|--------------|
| Llemma 7B | 7B | 41.0% | 18.6% | 20.6% | 22.1% |
| Llemma 34B | 34B | 64.6% | 26.3% | 21.0% | 21.3% |
| **DeepSeekMath-Base 7B** | **7B** | **66.9%** | **31.4%** | **25.8%** | **24.6%** |

**说明**: DeepSeekMath-Base 在工具辅助数学推理和形式化定理证明任务上均取得最佳结果。

### Table 4: NLU / 推理 / 代码能力

| Model | Size | MMLU | BBH | HumanEval | MBPP |
|-------|------|------|-----|-----------|------|
| Mistral 7B | 7B | 62.4% | 55.7% | 28.0% | 41.4% |
| DeepSeek-Coder-Base-v1.5 | 7B | 49.1% | 55.2% | 43.2% | 60.4% |
| **DeepSeekMath-Base 7B** | **7B** | **54.9%** | **59.5%** | **40.9%** | **52.6%** |

**说明**: 数学预训练提升了 MMLU 和 BBH 的推理能力，并通过保留代码 token 的训练维持了代码能力。

### Table 5: Instruct / RL 模型与其他模型对比

| Model | Size | GSM8K | MATH | MGSM-zh | CMATH |
|-------|------|-------|------|---------|-------|
| GPT-4 | - | 92.0% | 52.9% | - | 86.0% |
| Gemini Ultra | - | 94.4% | 53.2% | - | - |
| DeepSeek-LLM-Chat 67B | 67B | 84.1% | 32.6% | 74.0% | 80.3% |
| WizardMath-v1.1 7B | 7B | 83.2% | 33.0% | - | - |
| **DeepSeekMath-Instruct 7B** | **7B** | **82.9%** | **46.8%** | **73.2%** | **84.6%** |
| **DeepSeekMath-RL 7B** | **7B** | **88.2%** | **51.7%** | **79.6%** | **88.8%** |

**说明**: DeepSeekMath-RL 7B 超越所有 7B-70B 开源模型及多数闭源模型。GRPO 仅使用 GSM8K/MATH 的 CoT 数据就带来了跨基准的全面提升。

### Table 6: 代码训练对数学推理的影响

| 训练设置 | GSM8K | MATH | CMATH | GSM8K+Python | MATH+Python |
|----------|-------|------|-------|-------------|------------|
| General → Math | 19.1% | 14.4% | 37.2% | 14.3% | 6.7% |
| Code → Math (2-stage) | **21.9%** | **15.3%** | **39.7%** | 17.4% | 9.4% |
| Code & Math (1-stage) | 17.6% | 12.1% | 36.3% | **19.7%** | **13.5%** |

**说明**: 先代码训练后数学训练在无工具推理上最优，混合训练在工具辅助推理上最优且缓解了灾难性遗忘。

### Table 7: arXiv 论文对数学推理的影响

| 模型 | 语料 | GSM8K | MATH | OCW | SAT | MMLU STEM | CMATH | Gaokao MathCloze | Gaokao MathQA |
|------|------|-------|------|-----|-----|-----------|-------|-----------------|---------------|
| DeepSeek-Coder 7B | No Math Training | 29.0% | 12.5% | 6.6% | 40.6% | 38.1% | 45.9% | 5.9% | 21.1% |
| DeepSeek-Coder 7B | MathPile | 23.6% | 11.5% | 7.0% | 46.9% | 35.8% | 37.9% | 4.2% | 25.6% |
| DeepSeek-Coder 7B | ArXiv-RedPajama | 28.1% | 11.1% | 7.7% | 50.0% | 35.2% | 42.6% | 7.6% | 24.8% |

**说明**: arXiv 论文语料对数学推理没有显著提升，在某些基准上甚至造成性能下降。

### Table 10: 统一范式中的方法对比

| 方法 | 数据来源 | 奖励函数 | 梯度系数 |
|------|----------|----------|----------|
| SFT | $q, o \sim P_{sft}(Q, O)$ | - | 1 |
| RFT | $q \sim P_{sft}(Q), o \sim \pi_{sft}(O|q)$ | Rule | $\mathbb{I}(o)$ |
| DPO | $q \sim P_{sft}(Q), o^+, o^- \sim \pi_{sft}(O|q)$ | Rule | Equation 14 |
| Online RFT | $q \sim P_{sft}(Q), o \sim \pi_\theta(O|q)$ | Rule | $\mathbb{I}(o)$ |
| PPO | $q \sim P_{sft}(Q), o \sim \pi_\theta(O|q)$ | Model | $A_t$ |
| GRPO | $q \sim P_{sft}(Q), \{o_i\}_{i=1}^G \sim \pi_\theta(O|q)$ | Model | $\hat{A}_{i,t} + \beta(\frac{\pi_{ref}}{\pi_\theta} - 1)$ |

**说明**: 所有方法通过统一范式 $\nabla_\theta \mathcal{J} = \mathbb{E}[ \frac{1}{|o|} \sum_t GC \cdot \nabla_\theta \log \pi_\theta ]$ 理解，区别在于数据来源、奖励函数和梯度系数计算方式。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DeepSeekMath Corpus | 120B tokens | Common Crawl 多语言数学网页 | 预训练 |
| GSM8K | 8.5K | 小学数学应用题 | 评测 |
| MATH | 12.5K | 竞赛级数学（代数、几何、数论等） | 评测 |
| CMATH | - | 中文小学数学题 | 评测 |
| MGSM-zh | - | 多语言 GSM8K 中文版 | 评测 |
| SAT / OCW | - | 美国高考/大学课程数学 | 评测 |
| MMLU-STEM | - | 多学科科学问题 | 评测 |
| Gaokao-Math | - | 中国高考数学 | 评测 |
| miniF2F | - | 形式化奥林匹克数学 | 评测 |

### 实现细节

**预训练**:
- **初始化**: DeepSeek-Coder-Base-v1.5 7B
- **训练数据**: 500B tokens (56% 数学 + 4% AlgebraicStack + 10% arXiv + 20% GitHub 代码 + 10% 自然语言)
- **优化器**: AdamW ($\beta_1=0.9$, $\beta_2=0.95$, weight decay=0.1)
- **学习率**: 峰值 4.2e-4，multi-step schedule
- **Batch Size**: 10M tokens
- **上下文长度**: 4K

**SFT**:
- **数据量**: 776K 训练样本
- **训练步数**: 500 steps (batch size 256)
- **学习率**: 5e-5 (constant)
- **上下文长度**: 4K

**GRPO**:
- **学习率**: 1e-6
- **KL 系数**: 0.04
- **每问题采样数**: 64 个输出
- **最大长度**: 1024
- **Batch Size**: 1024
- **训练数据**: 144K GSM8K/MATH 问题的 CoT 格式数据

### 关键发现

1. **代码训练提升数学推理**: 先代码后数学的 2-stage 训练比纯数学训练更好，且代码 + 数学混合训练在工具使用推理上最优
2. **arXiv 论文无效**: 实验表明 arXiv 论文语料对数学推理没有显著帮助
3. **在线训练优于离线**: Online RFT 显著优于 RFT，策略模型实时采样数据更具优势
4. **GRPO 胜过 Online RFT**: GRPO 通过差异化梯度系数（正负样本不同强度）超越 Online RFT
5. **Process Supervision 优于 Outcome**: GRPO+PS 比 GRPO+OS 效果更好
6. **RL 提升 Maj@K 而非 Pass@K**: RL 增强的是输出分布的鲁棒性而非底层能力

---

## 批判性思考

### 优点
1. 建立了大规模数学预训练数据的系统性方法论，迭代式数据收集 pipeline 可推广到其他领域
2. GRPO 去掉了 critic 模型，显著降低 RL 训练资源需求，对开源社区有重要价值
3. 提供了统一范式分析多种训练方法，深入揭示了不同方法之间的本质联系
4. 实验设计全面，覆盖了预训练、SFT、RL 各阶段的大量消融实验

### 局限性
1. 几何和定理证明能力相对较弱（如三角形、椭圆相关题目处理不佳）
2. 在 few-shot 能力上远不如 GPT-4（零样本和少样本表现相近，而 GPT-4 能从少样本中显著受益）
3. 受限于 7B 模型规模，在某些复杂任务上仍有明显差距
4. arXiv 论文无效的结论有局限性（未测试特定数学子任务、未结合其他类型数据、未在更大规模上验证）

### 潜在改进方向
1. 在更复杂的 OOD 问题上探索 RL，结合 tree-search 等高级采样策略
2. 开发对噪声奖励信号鲁棒的 RL 算法（weak-to-strong alignment）
3. 提升 reward model 的泛化能力和不确定性建模
4. 构建高质量 process reward model 以提供更细粒度的训练信号

### 可复现性评估
- [x] 代码开源
- [x] 预训练模型
- [x] 训练细节完整
- [ ] 数据集可获取（DeepSeekMath Corpus 未公开）

---

## 关联笔记

### 基于
- [[DeepSeek-Coder]]: 作为初始化模型
- [[OpenWebMath]]: 作为种子语料
- [[PPO]]: GRPO 的基线和改进对象

### 对比
- [[Minerva]]: 540B 闭源数学模型，被 7B 模型超越
- [[Llemma]]: 最新的开源数学模型
- [[WizardMath]]: 使用 Evol-Instruct + PPO 的数学推理模型
- [[MetaMath]]: 使用数据增强的数学推理模型

### 方法相关
- [[GRPO]]: 核心创新
- [[Chain-of-Thought]]: 推理方法
- [[Program-of-Thought]]: 工具辅助推理
- [[Tool-Integrated Reasoning]]: 工具集成推理
- [[Rejection Sampling Fine-tuning]]: 对比方法
- [[Direct Preference Optimization]]: 对比方法

### 数据相关
- [[Common Crawl]]: 数据来源
- [[fastText]]: 分类器
- [[DeepSeekMath Corpus]]: 构建的数学语料库

---

## 速查卡片

> [!summary] DeepSeekMath
> - **核心**: 通过大规模数学预训练数据 (120B tokens) + GRPO 实现开源模型数学推理能力突破
> - **方法**: 迭代式 CC 数据收集 + 代码初始化 + 数学 SFT + GRPO RL
> - **结果**: MATH 51.7%（超越所有开源，逼近 GPT-4 52.9%）
> - **代码**: https://github.com/deepseek-ai/DeepSeek-Math

---

*笔记创建时间: 2026-07-10*