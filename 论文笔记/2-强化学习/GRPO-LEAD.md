---
title: "GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models"
method_name: "GRPO-LEAD"
authors: [Jixiao Zhang, Chunsheng Zuo]
year: 2025
venue: EMNLP 2025 (Main)
tags: [reinforcement-learning, grpo, mathematical-reasoning, reward-engineering, length-control, difficulty-aware-training]
zotero_collection: 2-强化学习
image_source: local
arxiv_html: https://arxiv.org/html/2504.09696
created: 2026-08-01
---

# 论文笔记：GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for Concise Mathematical Reasoning in Language Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Johns Hopkins University |
| 日期 | April 2025 |
| 项目主页 | https://github.com/aeroplanepaper/GRPO-LEAD |
| 对比基线 | [[DeepSeek-R1]], [[Light-R1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2504.09696) / [Code](https://github.com/aeroplanepaper/GRPO-LEAD) / [Model](https://huggingface.co/PlanePaper/LEAD-14B) |

---

## 一句话总结

> 针对 [[GRPO]] 在数学推理中奖励稀疏、冗长输出和难度不敏感三大缺陷，提出三项增强（长度奖励+显式惩罚+难度感知优势重加权），在 14B 模型上达到 SOTA。

---

## 核心贡献

1. **长度依赖的准确率奖励 (L)**: 用组内正确响应的标准化长度偏差做指数衰减，替代统一二元奖励，鼓励简洁推理
2. **错误答案显式惩罚 (E)**: 将错误回答奖励设为 -1（而非 0），使期望奖励仅在 $P(\text{correct}) > 0.5$ 时为正，抑制猜测行为
3. **难度感知优势重加权 (AD)**: 基于问题组内经验正确率 $\rho_q$ 的 logistic 函数重缩放优势值，放大难题学习信号
4. **系统性训练策略分析**: 研究了模型规模、SFT 数据质量、两阶段课程对 RL 效果的影响
5. **实验效率**: Stage1 仅 100 步（8xH20 24小时）即匹配或超越 Light-R1-14B-DS 的 Cons@32

---

## 问题背景

### 要解决的问题
[[Group Relative Policy Optimization|GRPO]] 是 R1 类推理模型的核心 RL 算法，但现有实现在数学推理中面临三个关键问题：
1. **奖励稀疏**: 二元准确率指标导致组内所有正确/错误响应的奖励完全一致，梯度信号弱
2. **输出冗长**: 模型利用训练预算生成长推理链（[[Reward Hacking|reward hacking]]），缺乏简洁性激励
3. **难度不敏感**: 所有问题统一对待，容易在简单题上过优化，忽视难题

### 现有方法的局限
- 余弦长度奖励 (cosine length reward) 和"黄金长度"目标 (golden length target) 使用静态启发式规则，对复杂多变的数学问题适应性差
- 无显式惩罚机制鼓励模型猜测而非严谨推理
- 无难度调整机制导致模型在简单题上浪费学习资源

### 本文的动机
通过三项互补增强 —— 动态组内长度奖励、显式错误惩罚、难度感知优势重加权 —— 系统性地解决 GRPO 的上述缺陷，同时保持训练效率。

---

## 方法详解

### 模型架构

GRPO-LEAD 不是新架构，而是对 **[[Group Relative Policy Optimization|GRPO]]** 训练框架的三项增强：

- **基础算法**: [[Group Relative Policy Optimization|GRPO]]（组内标准化优势 + PPO-style clipping）
- **基座模型**: [[DeepSeek-R1|DeepSeek-R1-Distilled-Qwen]]（7B/14B）
- **训练框架**: [[VERL]]
- **训练数据**: [[DeepScaler]] 数据集（约 40.3k 题），过滤难度 < 2.5 后剩余 ~9k 题
- **KL 惩罚**: 移除（$\beta = 0$），发现其抑制探索

### 核心模块

#### 模块1: 长度依赖准确率奖励 (Length-Dependent Accuracy Reward)

**设计动机**: 利用组内相对长度动态调整奖励，鼓励简洁推理而不牺牲正确性。

**具体实现**:
- 对每道题 $q$ 的 $G$ 个采样响应，提取**正确响应子集**，计算其 token 长度的均值 $\mu$ 和标准差 $\sigma$
- 对每个正确响应 $o$，计算标准化长度偏差 $z = \frac{|o| - \mu}{\sigma + \epsilon}$
- 使用指数衰减调制奖励：$R_{\text{accuracy}}(o|q) = \exp(-\alpha z)$，其中 $\alpha = 0.05$
- 短于均值的正确响应获得 $>1$ 的奖励，长于均值的获得 $<1$ 的奖励
- 错误响应：$R = 0$（仅长度奖励阶段）；$R = -1$（完整 LEAD）

**关键特性**: 动态适应每题难度分布，区别于静态长度约束。利用标准化偏差而非绝对长度。

#### 模块2: 显式错误惩罚 (Explicit Penalty for Incorrect Answers)

**设计动机**: 解决二元奖励下"猜测优于不答"的问题。

**具体实现**:
- 将错误回答的奖励从 $0$ 改为 $-1$
- 不施加长度惩罚的近似期望奖励：$\mathbb{E}[R] \approx 2P(\text{correct}) - 1$
- 仅当正确概率 > 0.5 时期望奖励为正，形成有原则的决策边界
- 鼓励模型"不确定时不猜"，提升 pass@1 和整体精度

#### 模块3: 难度感知优势重加权 (Difficulty-Aware Advantage Reweighting)

**设计动机**: 普遍应用奖励会偏向简单题，需放大难题学习信号。

**具体实现**:
- **难度估计**: $\rho_q = \frac{\text{该题正确响应数}}{\text{该题总响应数}}$（经验正确率，难题 $\rho_q$ 低）
- **重加权因子**: $w(\rho_q) = A + \frac{B - A}{1 + \exp[k(\rho_q - \rho_0)]}$，参数 $A=0.4, B=1.5, \rho_0=0.75, k=10$
- **难度感知优势**: $A_i' = \tilde{A}_i \cdot w(\rho_q)$ （正优势）或 $A_i' = \tilde{A}_i \cdot w(1-\rho_q)$ （负优势）
- 难题的低 $\rho_q$ 使正确回答获得 $w(\rho_q) \to B=1.5$ 的大权重更新
- 简单题的错误回答获得 $w(1-\rho_q) \to B=1.5$ 的大惩罚

#### 模块4: SFT 预训练与两阶段课程

**SFT 策略**:
- 从 [[DeepScaler]] 中选取 ~13k 题（难度 > 1），用 [[QwQ-32B]] 生成高质量分步解答
- SFT 初始表现不佳（可能过拟合），但 RL 后迅速恢复并超越

**14B 模型两阶段训练**:
- **Stage 1**: SFT → GRPO-LEAD 100 步（~9k 题），约 24h on 8xH20
- **Stage 2**: 筛选 Stage-1 准确率 ≤ 75% 的难题（~2,283 题）+ Light-R1 Stage 2 数据（~3,524 题），继续 240 步
- **n-gram 重复修复**: Stage 2 中临时移除长度奖励，对重复 n-gram 施加 -1.5 惩罚

---

## 关键公式

### 公式1: [[GRPO|GRPO 目标函数]]

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\; \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right)\right]
$$

**含义**: GRPO 的 PPO-style clipped 策略梯度目标，以组内标准化优势 $\hat{A}_{i,t}$ 替代 critic 网络

**符号说明**:
- $G$: 组数（不同 query 数）
- $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$: [[Importance Sampling|token 级重要性采样比]]
- $\hat{A}_{i,t}$: 组内标准化优势
- $\epsilon$: clipping 范围

### 公式2: [[Length-Dependent Accuracy Reward|长度依赖准确率奖励]]

$$
R_{\text{accuracy}}(o|q) = \begin{cases} \exp(-\alpha z), & \text{if } o \text{ is correct} \\ 0, & \text{if } o \text{ is incorrect} \end{cases}
$$

其中标准化长度偏差：

$$
z = \frac{|o| - \mu}{\sigma + \epsilon}
$$

**含义**: 正确响应按相对简洁度获得差异化奖励；$\mu$ 和 $\sigma$ 从组内正确响应子集计算

**符号说明**:
- $\mu$: 组内正确响应 token 数量均值
- $\sigma$: 组内正确响应 token 数量标准差
- $\epsilon$: 数值稳定小常数
- $\alpha = 0.05$: 长度惩罚强度

### 公式3: [[Explicit Penalty|含显式惩罚的完整奖励]]

$$
R_{\text{accuracy}}(o|q) = \begin{cases} \exp(-\alpha z), & \text{if } o \text{ is correct} \\ -1, & \text{if } o \text{ is incorrect} \end{cases}
$$

**含义**: 错误回答 -1 而非 0，使期望奖励在准确率 > 0.5 时才为正

**符号说明**:
- $\mathbb{E}[R] \approx 2P(\text{correct}) - 1$: 忽略长度惩罚时的近似期望奖励

### 公式4: [[Difficulty-Aware Advantage Reweighting|难度估计]]

$$
\rho_q = \frac{\text{number of correct responses for } q}{\text{total number of responses for } q}
$$

**含义**: 经验正确率作为难度逆代理。低 $\rho_q$ = 高难度

### 公式5: [[Difficulty-Aware Advantage Reweighting|Logistic 重加权因子]]

$$
w(\rho_q) = A + \frac{B - A}{1 + \exp[k(\rho_q - \rho_0)]}
$$

**含义**: 用 logistic 函数平滑映射难度到权重。在 $\rho_q = \rho_0$ 附近急剧变化

**符号说明**:
- $A = 0.4$: 权重下界（简单题最低权重）
- $B = 1.5$: 权重上界（难题最高权重）
- $\rho_0 = 0.75$: 拐点（75% 正确率处开始重加权）
- $k = 10$: 陡峭度（高值使过渡更尖锐）

### 公式6: [[Difficulty-Aware Advantage Reweighting|标准优势归一化]]

$$
\tilde{A}_i = \frac{R(o_i|q) - \mu_q}{\sigma_q + \epsilon}
$$

**含义**: 组内 reward 标准化得到优势

**符号说明**:
- $\mu_q$: 问题 $q$ 的组内平均 reward
- $\sigma_q$: 问题 $q$ 的组内 reward 标准差

### 公式7: [[Difficulty-Aware Advantage Reweighting|难度感知优势]]

$$
A_i' = \tilde{A}_i \cdot \begin{cases} w(\rho_q), & \text{if } \tilde{A}_i > 0 \\ w(1 - \rho_q), & \text{if } \tilde{A}_i \leq 0 \end{cases}
$$

**含义**: 正优势（好回答）在难题上被放大，负优势（差回答）在简单题上被放大

---

## 关键图表

### Figure 1: GRPO-LEAD 框架总览

![[GRPO-LEAD_fig1.png]]

**说明**: GRPO-LEAD 框架示意。左侧难题（2/8 正确，$\rho_q=0.25$，$w \approx 1.49$），右侧简单题（7/8 正确，$\rho_q=0.875$，$w \approx 0.65$）。展示长度正则化奖励、显式惩罚（-1）和难度权重如何共同影响策略更新。

### Figure 2: 验证集 Pass@1 训练曲线

![[GRPO-LEAD_fig2.png]]

**说明**: 7B 模型三步配置（GRPO / GRPO+L / GRPO+LAD）在 27 道验证题上的 Pass@1 收敛曲线。长度奖励（L）提供更丰富学习信号，优势重加权（AD）进一步加速收敛。验证集来自 AIMO2、CMU-MATH-AIMO 和 AIME24。

### Figure 3: Pass@1 vs 推理预算

![[GRPO-LEAD_fig3.png]]

**说明**: AIME24 (a) 和 AIME25 (b) 上不同推理长度预算下各消融配置的 Pass@1。GRPO+L 在低预算（<5k tokens）显著优于预训练基线。完整 LEAD 在 >5k tokens 预算下最优，因显式惩罚需要更多思考 tokens。

### Table 1: 7B 模型消融实验

| Ablation Setting | AIME24 Cons@32 | AIME24 Pass@1 | AIME24 Len$_{\text{avg}}$ | AIME25 Cons@32 | AIME25 Pass@1 | AIME25 Len$_{\text{avg}}$ |
|--------|---------|---------|---------|---------|---------|---------|
| Deepseek-7B | 0.767 | 0.431 | 6,990 | 0.467 | 0.292 | 7,113 |
| GRPO + len. reward | 0.767 | 0.438 | **5,275** | 0.533 | 0.308 | **5,210** |
| + adv. reweighting | 0.767 | 0.458 | 5,323 | 0.567 | 0.325 | 5,437 |
| + explicit penalty | **0.800** | **0.470** | 6,104 | **0.567** | **0.345** | 6,308 |

**说明**: 三项增强逐步提升 Pass@1 和 Cons@32。长度奖励大幅减少输出长度（~25%），显式惩罚用适度长度增长换取最高精度。

### Table 2: 14B 模型与基线对比

| Model Name | AIME24 Cons@32 | AIME24 Pass@1 | AIME24 Len$_{\text{avg}}$ | AIME25 Cons@32 | AIME25 Pass@1 | AIME25 Len$_{\text{avg}}$ |
|--------|---------|---------|---------|---------|---------|---------|
| DeepSeek-14B | 0.800 | 0.614 | 9,182 | 0.633 | 0.429 | 10,046 |
| Light-R1-14B-DS | 0.833 | 0.641 | 9,571 | 0.767 | 0.505 | 10,194 |
| LEAD-stage1 | 0.833 | 0.629 | 8,790 | 0.767 | 0.523 | 9,371 |
| **LEAD-stage2** | **0.867** | **0.650** | **8,267** | **0.767** | **0.539** | **8,668** |

**说明**: LEAD-stage2 在 AIME24 Cons@32 超越 Light-R1 4%，同时输出减少 1,300+ tokens。Stage1 仅 100 步即匹配 Light-R1 Cons@32 且输出更短。

### Table 3: LiveCodeBench 编码能力

| Model | Accuracy | Avg. Tokens (Overall) | Easy | Medium | Hard |
|--------|---------|---------|---------|---------|---------|
| LEAD-14B | **0.5156** | 6322 | 3998 | 6912 | 8000 |
| DeepSeek-R1-Distill-Qwen-14B | 0.5103 | 5794 | 3046 | 6429 | 7856 |

**说明**: 编码能力轻微提升，但输出更长（数学训练的长度压缩未泛化到代码）。

### Table 4: AIME25 难度分层分析

| Model | Cons@32 | Avg. Correct | Avg. Answer | Precision | Pass@1 |
|--------|---------|---------|---------|---------|---------|
| **Normal (1-5)** | | | | | |
| Deepseek-7B | 0.8 | 18.8 | 20.3 | 0.708 | 0.588 |
| GRPO + LAD | 0.9 | 20.1 | 26.9 | 0.687 | 0.628 |
| **GRPO + LEAD** | 0.8 | 22.0 | 24.5 | **0.723** | **0.688** |
| **Difficult (6-10)** | | | | | |
| Deepseek-7B | 0.4 | 8.3 | 13.8 | 0.404 | 0.259 |
| GRPO + LAD | 0.6 | 9.8 | 24.2 | 0.448 | **0.306** |
| GRPO + LEAD | 0.6 | 9.7 | 20.0 | 0.421 | 0.303 |
| **Highly Difficult (11-15)** | | | | | |
| Deepseek-7B | 0.2 | 0.9 | 2.0 | 0.230 | 0.028 |
| GRPO + LAD | 0.2 | 1.3 | 14.6 | 0.156 | 0.041 |
| GRPO + LEAD | **0.4** | 1.5 | 7.7 | **0.355** | **0.047** |

**关键发现**: 优势重加权在"困难"层（6-10）带来最大相对增益（Pass@1 +13.7%）。在"极难"层（11-15），GRPO+LAD 与 GRPO+L 持平（Pass@1 均为 0.041），但加入显式惩罚后提升至 0.047。精度在极难题上从 0.156 跃升至 0.355。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DeepScaler]] | ~40.3k 题 | AIME + AMC + Omni-MATH + Still 数据集 | RL 训练（过滤难度 <2.5 后约 9k 题） |
| SFT 数据 | ~13k 题 | DeepScaler 难度 >1 的子集 + QwQ-32B 解答 | 14B SFT 预训练 |
| Stage 2 数据 | ~3,524 题 | Stage-1 准确率 ≤75% 的难题 + Light-R1 难题 | 14B Stage 2 RL |
| AIME24 / AIME25 | 各 30 题 | 高难度数学竞赛题 | 主要评测 |
| LiveCodeBench | 880 题 (v5) | 代码生成 | 域外泛化测试 |

### 实现细节

- **基座模型**: DeepSeek-R1-Distilled-Qwen-7B-Math / DeepSeek-R1-Distilled-Qwen-14B
- **训练框架**: [[VERL]]
- **优化器**: Adam，学习率 $1 \times 10^{-6}$
- **Batch Size**: 32
- **Group Size**: 8（每题 8 个 rollout）
- **温度**: 0.6
- **KL 系数**: 0（移除 KL 惩罚）
- **生成长度限制**: 7B 模型 8k tokens / 14B 模型 14k tokens
- **硬件**: 8 x H20 GPU；Stage 1 约 24 小时（100 步）
- **超参数**: $\alpha=0.05$, $A=0.4$, $B=1.5$, $\rho_0=0.75$, $k=10$, n-gram 重复惩罚 -1.5

### 关键发现

1. **长度奖励降低 25% 输出**: 7B 模型上 AIME24 从 6,990 降至 5,275 tokens（-24.5%），AIME25 从 7,113 降至 5,210（-26.8%），且 Pass@1 不降反升
2. **100 步 = SOTA**: LEAD-stage1 仅 100 步（24h on 8xH20）即匹配 Light-R1-14B-DS
3. **SFT 先降后升**: SFT 初始精度下降（过拟合），但 RL 后快速恢复并超越直接 RL
4. **KL 惩罚有害**: 实验发现 KL 项抑制探索，移除后效果更好
5. **代码泛化有限**: 数学训练的长度压缩未泛化到代码任务

---

## 批判性思考

### 优点
1. **方法论系统且优雅**: 三项增强（L/E/AD）各自解决 GRPO 的一个明确缺陷，组合后形成互补。每个组件都有清晰的数学动机和预期效果
2. **实用效率突出**: Stage1 仅需 100 步/24h 即可匹敌 SoTA，训练成本极低，对学术界友好
3. **实验设计扎实**: 消融实验清晰展示每项增强的增量贡献，包括长度-精度权衡、推理预算消融、难度分层分析
4. **代码和模型完全开源**: GitHub + HuggingFace 发布训练代码、SFT 数据集（GRPO-LEAD-SFTData，~12k 样本）、7B 和 14B checkpoint
5. **定性分析有说服力**: 附录中对比 GRPO+L 与 Deepseek-7B 的具体输出，展示了长度奖励在结构、冗余、语言简洁度上的实质改进

### 局限性
1. **仅验证数学推理**: 所有主要实验在数学领域，代码泛化仅以单次 LiveCodeBench 测试，未见 GPQA/逻辑推理的评估
2. **计算资源受限**: 作者坦言无法进行超参搜索、缺少部分消融曲线、baseline checkpoint 丢失，部分结论证据不够充分
3. **超参敏感性未分析**: $\alpha, A, B, \rho_0, k$ 五个超参联合调节，但未做敏感性分析，读者难以判断这些参数在其他场景下是否需要大量调整
4. **Group Size = 8 的固定性**: 所有实验使用 G=8，但 group size 对 GRPO 类的组内标准化影响显著，缺少不同 G 值下的行为分析
5. **KL=0 的理由不够充分**: 只是引用其他工作说"KL 抑制探索"，但移除 KL 的长期影响（如奖励过度优化）未被探讨
6. **域外泛化**: 代码任务上模型输出更长而非更短，说明长度压缩机制具有强任务依赖性

### 潜在改进方向
1. **域泛化研究**: 测试 GRPO-LEAD 在 GPQA、逻辑推理、甚至通用对话任务上的效果
2. **自适应超参**: 用元学习或在线调参替代固定超参，提升跨任务泛化性
3. **与 REWARD SHAPING 理论结合**: 用势函数理论（potential-based reward shaping）解释为什么这组修改是策略不变的
4. **更大规模扩展**: 测试 32B/72B 模型上 GRPO-LEAD 的效果，验证 scaling 行为
5. **多模态推理**: 将方法扩展到几何/图表数学推理任务

### 可复现性评估
- [x] 代码开源
- [x] 预训练模型
- [x] 训练细节完整
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[Group Relative Policy Optimization|GRPO]]: GRPO-LEAD 的核心基础算法
- [[DeepSeek-R1]]: 首次大规模使用 GRPO 训练的推理模型
- [[PPO]]: GRPO clipping 机制的前身
- [[DeepSeekMath]]: 首次提出 GRPO 的工作
- [[Light-R1]]: 主要对比基线，使用余弦长度奖励
- [[DeepScaler]]: RL 训练数据来源

### 对比
- [[DAPO]]: 另一种 GRPO 改进（非对称 clipping + 动态采样），对比不同改进方向
- [[GP-CPO|GPG]]: GPG/GP-CPO 类方法，也与 GRPO 比较
- [[RLVR]]: 可验证奖励 RL 的通用框架
- [[Reward Shaping|奖励塑形]]: 长度奖励和显式惩罚都可视为奖励塑形特例

### 方法相关
- [[Length-Dependent Accuracy Reward]]: 核心方法组件
- [[Difficulty-Aware Advantage Reweighting]]: 核心方法组件
- [[Explicit Penalty]]: 核心方法组件
- [[Advantage Normalization]]: GRPO 的基础操作
- [[Reward Sparsity]]: 要解决的核心问题

### 硬件/数据相关
- [[DeepScaler]]: 训练数据集
- [[AIME]]: 主要评测基准
- [[VERL]]: 训练框架
- [[QwQ-32B]]: SFT 数据生成模型

---

## 速查卡片

> [!summary] GRPO-LEAD: A Difficulty-Aware RL Approach for Concise Math Reasoning
> - **核心**: 三项 GRPO 增强（长度奖励 + 显式惩罚 + 难度重加权）系统性解决奖励稀疏、冗长输出和难度不敏感
> - **方法**: 组内标准化长度偏差指数衰减奖励 + -1 错误惩罚 + logistic 难度权重缩放优势
> - **结果**: 14B SOTA (AIME24 Cons@32=0.867, Pass@1=0.650)，7B 消融 Pass@1 从 0.431 提升至 0.470，输出长度减少 ~25%
> - **代码**: https://github.com/aeroplanepaper/GRPO-LEAD

---

*笔记创建时间: 2026-08-01*
