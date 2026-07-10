---
title: "Hierarchical Sparse Attention Done Right: Toward Infinite Context Modeling"
method_name: "HiLS-Attention"
authors: [Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang Li, Lei Zhu, Yan Wang, Sirui Han, Yushi Bai, Kewei Tu, Haitao Mi, Leo Liang]
year: 2026
venue: arXiv
tags: [sparse-attention, long-context, llm-architecture, length-extrapolation, efficient-attention, hierarchical-attention]
zotero_collection: 3-大模型架构
image_source: online
arxiv_html: https://arxiv.org/html/2607.02980v1
created: 2026-07-10
---

# 论文笔记：Hierarchical Sparse Attention Done Right

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tencent HY Team, ShanghaiTech University, HKUST, UCSD |
| 日期 | July 2026 |
| 代码 | [GitHub](https://github.com/Tencent-Hunyuan/HiLS-Attention) |
| 对比基线 | [[Native Sparse Attention|NSA]], [[DashAttention]], [[InfLLM]], [[HSA-UltraLong]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.02980) / [HTML](https://arxiv.org/html/2607.02980v1) |

---

## 一句话总结

HiLS-Attention 通过层次化 softmax 分解实现端到端可学习的块稀疏注意力，在保持全注意力性能的同时实现超过 64 倍训练长度的外推。

---

## 核心贡献

1. **端到端可学习的稀疏检索**: 提出 HiLS-Attention，一种基于层次化 softmax 的原生稀疏注意力机制，使块选择可直接通过语言建模损失进行端到端训练。
2. **块质量的线性代理理论**: 证明有效块摘要应与全注意力导出的块质量的泰勒一阶展开在数学上对齐，从而为准确块选择提供充分表示能力。
3. **低成本模型转换**: 全注意力模型可通过轻量级继续预训练（仅需 50B tokens）转换为 HiLS-Attention，保留短上下文性能的同时在长上下文任务上超越全注意力。

---

## 问题背景

### 要解决的问题
[[Large Language Model|大语言模型]] 在处理长序列时面临两个核心挑战：[[Softmax Attention|全注意力]] 的 $O(N^2)$ 计算复杂度和糟糕的长度外推能力。

### 现有方法的局限
现有的块稀疏注意力方法（如 [[Native Sparse Attention|NSA]]、[[DashAttention]]、[[InfLLM]]）虽然通过选择性关注相关上下文块来保持恒定计算成本，但**块选择不准确**导致性能始终不及全注意力。根本原因在于：
- 非参数化块摘要（如 mean-pooling）表达力有限，会丢失关键信息
- 参数化摘要仅用于为块打分，在 hard top-K 选择后摘要和分数被丢弃
- [[Language Modeling|语言建模]]损失无法直接优化块选择，导致选择不准确

### 本文的动机
观察到块选择准确性的关键在于：块摘要应具有足够表达力，且可通过 LM 损失端到端训练。从 [[Naive Block Sparse Attention|Naive BSA]] 出发，推导出一个可学习的块质量代理，并通过层次化分解使检索分数参与前向注意力计算。

---

## 方法详解

### 模型架构

HiLS-Attention 采用 **解码器 Transformer + 层次化稀疏注意力** 架构：
- **输入**: Token 序列 $\bm{x} = \{x_1, x_2, \ldots, x_N\}$
- **分区**: 将序列划分为大小为 $S$ 的非重叠块
- **核心机制**: [[Hierarchical Softmax|层次化 softmax]] 将注意力分解为块间（inter-chunk）和块内（intra-chunk）两个阶段
- **滑动窗口**: 保留大小为 $W$ 的局部滑动窗口（实验中 $W=512$）
- **选中块数**: 全局检索 top-$K$ 个远端块（实验中 $K=32$）
- **额外参数**: 仅占模型总参数的 0.6%（landmark token + low-rank Q-Cal）

### 核心模块

#### 模块1: 块质量估计（Chunk Mass Surrogate）

**设计动机**: Naive BSA 需要计算全注意力才能得到精确块质量 $Z_{i,c} = \sum_{j\in\mathcal{T}_c}\exp(s_{i,j})$，这完全消除了稀疏注意力的计算优势。需要一种不计算所有 token 对分数的块质量估计方法。

**具体实现**:
- 在每个块末尾添加一个特殊的 [[Landmark Token]]，用于生成块级别的查询向量 $\mathbf{q}'_c$
- 块摘要键 $\mathbf{k}'_c$ 由 landmark token 的 key 和 bias $b'_c$ 组成
- 块质量的线性代理通过 $\mathbf{q}'_c$ 的一阶 [[Taylor Expansion|泰勒展开]] 推导得出：

**数学推导**: 块质量的 LogSumExp 可表示为 $\log Z_{i,c} = \max_j(s_{i,j}) + \log\sum_j\exp(s_{i,j} - \max_j(s_{i,j}))$。通过用 $\mathbf{q}'_c$ 替代 $\mathbf{q}_i$，得到代理分数 $\hat{s}_{i,c} = \frac{\mathbf{q}_i^\top\mathbf{k}'_c}{\sqrt{d}} + b'_c$，并定义 $\hat{Z}_{i,c} = \exp(\hat{s}_{i,c})$。

#### 模块2: 层次化 Softmax（Hierarchical Factorization）

**设计动机**: 使代理质量 $\hat{Z}_{i,c}$ 参与前向注意力计算，从而使 LM 损失的梯度能直接反向传播到块摘要表示。

**具体实现**:
- 将标准 softmax 权重分解为两个因子的乘积：
  - **块内 softmax** (intra-chunk): $\frac{\exp(s_{i,j})}{Z_{i,c(j)}}$ — 块内 token 的相对重要性
  - **块间 softmax** (inter-chunk): $\frac{\hat{Z}_{i,c(j)}}{\hat{\mathcal{Z}}_i}$ — 块级别的相对重要性
- 整体权重：$w_{i,j} = \frac{\exp(s_{i,j})}{Z_{i,c(j)}} \times \frac{\hat{Z}_{i,c(j)}}{\hat{\mathcal{Z}}_i}$
- 各 token 先在块内局部归一化，再通过学习的块质量进行全局融合
- 滑动窗口内的 token 按 $Z_{i,\text{swa}}$ 处理

#### 模块3: Low-Rank Query Calibration (Q-Cal)

**设计动机**: 原始 token 级查询 $\mathbf{q}_i$ 可能不是估计块级质量的最佳选择，因为块摘要 $\mathbf{k}'_c$ 是多个 token 的压缩表示。

**具体实现**:
- 引入轻量级低秩适配模块：$\Delta\mathbf{q}_i = \mathbf{W}^{\text{up}}\mathbf{W}^{\text{down}}\mathbf{h}_i$
- 校准后查询：$\hat{\mathbf{q}}_i = \mathbf{q}_i + \Delta\mathbf{q}_i$
- 校准后分数：$\hat{s}_{i,c} = \frac{\hat{\mathbf{q}}_i^\top\mathbf{k}'_c}{\sqrt{d}} + b'_c$
- $\mathbf{W}^{\text{up}} \in \mathbb{R}^{d \times r}$, $\mathbf{W}^{\text{down}} \in \mathbb{R}^{r \times d_{\text{model}}}$，其中 $r \ll d_{\text{model}}$

#### 模块4: 硬件高效 Kernel 设计

**设计动机**: 不同 token 对应不同块集合，朴素实现无法获得加速效果且可能导致内存爆炸。需要硬件-软件协同设计。

**具体实现**:
- 将 $M$ 个相邻查询 token 分组，取它们所选块的并集
- Tensor Core 计算形状从 NSA 的 $(G, d) \times (d, S)$ 变为 $(M \times G, d) \times (d, S)$
- 高效 Tensor Core 利用只需 $M \times G \geq 16$（而不是 NSA 要求的 $G \geq 16$）
- 相邻 token 块重叠率高达 80%，打包后减少冗余内存访问

#### 模块5: GQA 适配

**设计动机**: 现代 LLM 常使用 [[Grouped-Query Attention|GQA]]，同一组内的查询头应共享相同的检索块集合以利用批处理 kernel。

**具体实现**:
- 对组内每个查询头分别计算归一化块权重
- 在组内取最大值聚合（max over heads）
- 使用组级分数选择 top-$K$ 块
- 确保一个块只要对组内任意头重要就会被选中

#### 模块6: 继续训练策略

**设计动机**: 将现有全注意力模型高效转换为 HiLS-Attention，同时保留原有能力。

**具体实现**:
- **Landmark Token Tuning**: 冻结基座模型所有参数，仅训练 landmark token 嵌入和 Q-Cal 投影矩阵（不到 1% 参数），训练不超过 5B tokens
- **Full-Parameter Tuning**: 全参数微调，同时更换位置编码为 [[HoPE]] 以最大化长度泛化

---

## 关键公式

### 公式1: [[Chunk Mass|块质量定义]]

$$
Z_{i,\text{swa}} = \sum_{j=\ell(i)}^{i} \exp(s_{i,j}), \quad
Z_{i,c} = \sum_{j \in \mathcal{T}_c} \exp(s_{i,j})
$$

**含义**: 定义滑动窗口区域和远端块的注意力质量（attention mass），即块内 token 指数化注意力分数的总和。

**符号说明**:
- $s_{i,j} = \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d}$: token 间注意力 logit
- $\ell(i) = \lfloor (i-W+1)/S \rfloor S$: 对齐到块边界的滑动窗口左边界
- $\mathcal{T}_c$: 第 $c$ 个块中的 token 索引集合

### 公式2: [[Top-K Selection|块选择]]

$$
\bm{\mathcal{I}}_i = \{c \in \mathcal{C}_i \mid \operatorname{rank}_\downarrow(Z_{i,c}) < K\}
$$

**含义**: 选择块质量 $Z_{i,c}$ 排名前 $K$ 的块。

**符号说明**:
- $\mathcal{C}_i = \{0, 1, \ldots, \frac{\ell(i)}{S} - 1\}$: 候选块索引集
- $\operatorname{rank}_\downarrow$: 降序排名

### 公式3: [[Naive Block Sparse Attention|Naive BSA 输出]]

$$
w_{i,j} = \begin{cases}
\frac{\exp(s_{i,j})}{\mathcal{Z}_i}, & j \text{ is selected} \\
0, & \text{otherwise}
\end{cases}, \quad
\mathbf{o}_i = \sum_{j \leq i} w_{i,j} \mathbf{v}_j
$$

**含义**: 仅对选中的 token（滑动窗口 + top-K 块）计算 softmax 归一化注意力输出，未选中 token 权重为零。

### 公式4: [[LogSumExp|LogSumExp 块质量]] 的分解

$$
\log Z_{i,c} = \max_{j \in \mathcal{T}_c} s_{i,j} + \log \sum_{j \in \mathcal{T}_c} \exp(s_{i,j} - \max_{j \in \mathcal{T}_c} s_{i,j})
$$

**含义**: 将 $\log Z_{i,c}$ 分解为最大值项和剩余项之和，为线性代理提供理论基础。

### 公式5: [[Proposition 3.1|块质量的一阶泰勒代理]]

$$
\log Z_{i,c} = \max_{j \in \mathcal{T}_c} \mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d} + \text{remainder} \approx \mathbf{q}_i^\top \mathbf{k}'_c / \sqrt{d} + b'_c
$$

**含义**: 块质量的对数可通过查询与块摘要键的点积加偏置项来近似，实现线性代理。

### 公式6: [[Query Summary Key|查询摘要键构造]]

$$
\mathbf{k}'_c \in \mathbb{R}^d, \quad b'_c \in \mathbb{R}
$$

**含义**: 通过 landmark token 的查询向量 $\mathbf{q}'_c$ 构造块摘要，形成对块内所有 key 的压缩表示和熵校准。

**符号说明**:
- $(\mathbf{k}'_c, b'_c)$: 从 landmark token 查询推导的压缩键和偏置

### 公式7: [[Linear Chunk Scoring|线性块评分]]

$$
\hat{s}_{i,c} = \frac{\mathbf{q}_i^\top \mathbf{k}'_c}{\sqrt{d}} + b'_c, \quad
\bm{\mathcal{I}}_i = \{c \in \mathcal{C}_i \mid \operatorname{rank}_\downarrow(\hat{s}_{i,c}) < K\}
$$

$$
\hat{Z}_{i,c} = \exp(\hat{s}_{i,c}), \quad
\hat{\mathcal{Z}}_i = \sum_{c \in \mathcal{I}_i} \hat{Z}_{i,c} + Z_{i,\text{swa}}
$$

**含义**: 使用线性代理分数进行 top-K 块选择，将选择的块质量估计为指数化代理分数之和。

### 公式8: [[Hierarchical Softmax Factorization|层次化注意力分解]]

$$
w_{i,j} = \frac{\exp(s_{i,j})}{\mathcal{Z}_i} = \underbrace{\frac{\exp(s_{i,j})}{Z_{i,c(j)}}}_{\text{intra-chunk}} \times \underbrace{\frac{Z_{i,c(j)}}{\mathcal{Z}_i}}_{\text{inter-chunk}} \approx \frac{\exp(s_{i,j})}{Z_{i,c(j)}} \times \underbrace{\frac{\hat{Z}_{i,c(j)}}{\hat{\mathcal{Z}}_i}}_{\text{surrogate}}
$$

**含义**: 将注意力权重分解为块内归一化项和块间质量项，其中块间质量用可学习的代理替代，使得检索分数可通过 LM 损失端到端优化。

### 公式9: [[Low-Rank Query Calibration|低秩查询校准]]

$$
\Delta\mathbf{q}_i = \mathbf{W}^{\text{up}}\mathbf{W}^{\text{down}}\mathbf{h}_i, \quad
\hat{\mathbf{q}}_i = \mathbf{q}_i + \Delta\mathbf{q}_i, \quad
\hat{s}_{i,c} = \frac{\hat{\mathbf{q}}_i^\top \mathbf{k}'_c}{\sqrt{d}} + b'_c
$$

**含义**: 通过低秩适配模块（残差连接）校准 token 查询，使其更适用于块级质量估计。

**符号说明**:
- $\mathbf{W}^{\text{up}} \in \mathbb{R}^{d \times r}$, $\mathbf{W}^{\text{down}} \in \mathbb{R}^{r \times d_{\text{model}}}$: 低秩投影矩阵
- $d$: 注意力头维度， $d_{\text{model}}$: 模型隐藏维度， $r$: 低秩瓶颈维度

---

## 关键图表

### Figure 1: 实验结果概览

![Figure 1](https://arxiv.org/html/2607.02980v1/x1.png)

**说明**: HiLS-Attention 在仅 50B 继续训练 token 后的三大优势。(a) 在 YaRN 扩展的 4 倍长度之外仍然保持强大超长上下文外推能力；(b) 推理速度更快；(c)(d) 在原始训练长度和 YaRN 外推范围内，对短/中上下文任务的表现与全注意力相当。

### Figure 2: 上下文内检索结果

![Figure 2](https://arxiv.org/html/2607.02980v1/x5.png)

**说明**: 现有稀疏注意力方法与全注意力在上下文检索任务上的差距。HiLS-Attention 是唯一达到完美域内 NIAH 性能的原生稀疏注意力方法。

### Figure 3: HiLS-Attention 概览

![Figure 3](https://arxiv.org/html/2607.02980v1/data/HiLS-Attn.png)

**说明**: HiLS-Attention 的整体架构。Naive BSA 需要计算全 QK 来获取精确块质量 $Z_c$ 以选择 top-K 块。HiLS-Attention 使用压缩块键 $\mathbf{k}'_c$ 高效估计块质量代理 $Z'_c \propto \exp(\mathbf{q}^\top \mathbf{k}'_c)$。注意力分解为两个阶段：**块间 softmax**（指定分配给每个块的注意力总质量）和**块内 softmax**（将每个块的注意力质量分配到其 token 中）。

### Figure 4: Kernel 设计对比

| (a) NSA kernel | (b) HiLS-Attention kernel |
|---|---|
| ![NSA kernel](https://arxiv.org/html/2607.02980v1/x6.png) | ![HiLS kernel](https://arxiv.org/html/2607.02980v1/x7.png) |

**说明**: (a) NSA 每次处理一个查询 token，Tensor Core 运算形状为 $(G, d) \times (d, S)$，需要 $G \geq 16$ 才能高效利用。(b) HiLS-Attention 打包 $M$ 个相邻查询 token，取所选块并集，运算形状为 $(M \times G, d) \times (d, S)$，只需 $M \times G \geq 16$，大幅降低 GQA 组大小要求。

### Figure 5: 1.4B 模型的训练动态

![Figure 5](https://arxiv.org/html/2607.02980v1/x8.png)

**说明**: 1.4B 模型在不同训练步数的 Perplexity (a) 和 RULER 准确率 (b)。左侧：Full-Attention with RoPE；右侧：HiLS-Attention with HoPE。HiLS-Attention 在训练 300B token 后不仅追平全注意力困惑度，而且在 RULER 上表现更优。

### Figure 6: 推理效率分析

![Figure 6](https://arxiv.org/html/2607.02980v1/x9.png)

**说明**: HiLS-Attention 的推理效率分析，展示其稀疏 KV 访问和计算带来的推理加速。

### Figure 7: 块重叠分析

![Figure 7](https://arxiv.org/html/2607.02980v1/x10.png)

**说明**: 相邻查询 token 的块选择重叠率分析，验证了打包策略的合理性。

### Table 1: 345M 模型 Perplexity（8K 训练长度）

| Model | Extra Param | 64 | 128 | 512 | **8K** | 32K | 128K | 512K |
|-------|-------------|------|------|------|------|------|------|------|
| Full-Attn RoPE | – | 33.92 | 26.89 | 18.68 | 4.96 | >10² | – | – |
| Full-Attn HoPE | – | 34.15 | 26.97 | 18.73 | 4.95 | 6.42 | >10² | – |
| Full-Attn HoPE (+MLP) | 0.6% | 34.28 | 27.12 | 18.76 | 4.95 | 5.85 | >10² | – |
| SWA-RoPE | – | 35.38 | 27.94 | 19.30 | 8.95 | 9.01 | 8.44 | 8.47 |
| NSA-RoPE | 0.2% | 34.15 | 27.11 | 18.85 | 5.01 | 7.62 | 11.75 | 19.14 |
| Dash-Attention-RoPE | 0.006% | 33.70 | 26.74 | 18.67 | 5.00 | >10² | – | – |
| HSA-Ultralong | 12.7% | 33.19 | 26.26 | 21.50 | 8.81 | 5.84 | 4.99 | 4.54 |
| Naive-BSA-HoPE | – | 36.12 | 29.06 | 20.45 | 4.94 | 3.94 | 8.67 | OOM |
| **HiLS-Attn-HoPE** | **0.6%** | **33.97** | **26.91** | **18.65** | **4.94** | **4.34** | **4.71** | **5.95** |

**说明**: HiLS-Attention 在 8K 域内达到与全注意力相当的困惑度（4.94），在 32K 外推时（4.34）甚至优于全注意力。远超其他稀疏注意力方法的外推能力。

### Table 2: 345M 模型 RULER 结果

| Model | 8K S-N | 8K MK-MQ | 8K VT | 16K S-N | 16K MK-MQ | 16K VT | 32K S-N | 32K MK-MQ | 32K VT | 128K S-N | 128K MK-MQ | 128K VT | 512K S-N | 512K MK-MQ | 512K VT |
|-------|--------|----------|-------|---------|-----------|--------|---------|-----------|--------|----------|------------|--------|----------|------------|--------|
| Full-Attn RoPE | 100 | 97 | 34 | 0 | 0 | 0 | - | - | - | - | - | - | - | - | - |
| SWA-RoPE | 88 | 91 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NSA-RoPE | 88 | 87 | 33 | 20 | 26 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Dash-Attn | 96 | 75 | 28 | 1 | 2 | 1 | 0 | 0 | 0 | - | - | - | - | - | - |
| **HiLS-Attn-HoPE** | **100** | **100** | **99** | **100** | **100** | **99** | **100** | **100** | **100** | **100** | **100** | **98** | **100** | **100** | **91** |

**说明**: HiLS-Attention 在域内（8K）达到完美 NIAH 性能（100%）。在 512K 超长外推下仍保持 91%+ 准确率。HiLS 是唯一在所有长度和任务类型上都表现出色的原生稀疏注意力方法。S-N: Single Needle, MK-MQ: Multi-Key Multi-Query, VT: Variable Tracking。

### Table 3: 345M 模型 Perplexity（256K 继续训练）

| Model | 256K (domain) | 512K |
|-------|:------:|:------:|
| Full-Attn (YaRN) | 3.69 | 25.14 |
| SWA-RoPE | 18.43 | >10² |
| NSA-RoPE | 60.03 | OOM |
| **HiLS-Attn-HoPE** | **3.80** | **3.90** |

**说明**: 在 256K 继续训练后，HiLS-Attention 在域内与全注意力相当（3.80 vs 3.69），但外推到 512K 时全注意力严重退化（25.14），而 HiLS 几乎不变（3.90）。

### Table 4: 345M RULER 结果（256K 训练）

| Model | 256K S-N | 256K MK-MQ | 256K VT | 512K S-N | 512K MK-MQ | 512K VT |
|-------|:--------:|:----------:|:--------:|:--------:|:----------:|:--------:|
| Full-Attn (YaRN) | 100 | 100 | 26 | 0 | 0 | 0 |
| **HiLS-Attn-HoPE** | 100 | 100 | **76** | 100 | 100 | **59** |

**说明**: 256K 训练后，HiLS-Attention 在 Variable Tracking 任务上大幅超越全注意力（76% vs 26%），且在外推时保持显著优势（59% vs 0%）。

### 其他关键结果

**1.4B 模型下游评估**:

| Benchmark | Full-Attn RoPE | HiLS-Attn 8K (300B) | HiLS-Attn 256K (CPT) |
|-----------|:--------------:|:-------------------:|:--------------------:|
| PPL (4K) | 11.31 | 11.42 | 11.53 |
| ARC-C | 33.53 | 33.28 | 33.53 |
| ARC-E | 65.53 | 63.47 | 63.72 |
| HellaSwag | 44.93 | 44.47 | 44.14 |
| Lambada | 56.33 | 56.35 | 55.17 |
| PIQA | 71.30 | 71.62 | 71.54 |
| RULER (NIAH) | 60.98 | 95.36 | 94.85 |

**7B LongBench 结果**: HiLS-Attention 在 50B 继续训练后在 LongBench 上超越全注意力基线，加权整体 33.8 vs 32.8，在 >32K 长度段优势更明显（34.7 vs 28.2）。

---

## 实验

### 数据集
- **预训练**: C4 / RedPajama 等标准预训练语料
- **长上下文评估**: [[RULER]]（合成检索任务）、[[LongBench]]（真实长上下文基准）
- **短上下文评估**: ARC-C, ARC-E, HellaSwag, Lambada, PIQA

### 实现细节

- **345M 模型**: 遵循 GPT-2 Medium 架构，8K 上下文长度训练
- **1.4B 模型**: 更大规模的训练，300B token 从头训练
- **7B 模型**: 从全注意力基座模型继续训练 50B tokens
- **稀疏配置**: 滑动窗口 512，块大小 64，top-K 32（总 2K token 预算）
- **位置编码**: [[HoPE]]（混合 RoPE + NoPE）
- **优化器**: AdamW，cosine learning rate schedule
- **硬件**: 详细配置见附录 E

### 可视化结果

HiLS-Attention 在性能上可媲美甚至超越全注意力，同时实现：
- 更长的有效上下文外推（高达 512 倍训练长度）
- 更高效的推理（稀疏 KV 访问和计算）
- Variable Tracking 任务上比全注意力高 50% 准确率

---

## 批判性思考

### 优点
1. **端到端可学习性**: 首次实现原生稀疏注意力的端到端检索学习，块选择可直接通过 LM 损失优化
2. **极致外推能力**: 以 8K 训练长度实现 4M 上下文（512 倍外推），90% 检索准确率
3. **轻量级转换**: 全注意力模型仅需 50B token 继续训练即可转换为 HiLS-Attention
4. **硬件友好设计**: 创新的打包 kernel 设计降低了对大 GQA 组大小的依赖
5. **理论支撑**: 提供了块质量代理与泰勒展开的理论联系，有坚实的数学基础

### 局限性
1. **位置编码依赖**: 只有配合 HoPE 而非标准 RoPE 时才能充分发挥性能
2. **额外参数开销**: 0.6% 额外参数（landmark + Q-Cal），虽然很小但仍为增量
3. **块选择二次项**: 路由开销为 $O(N^2/S)$，在极长序列上可能成为瓶颈
4. **Landmark Token 工程开销**: 实现需要特殊 token 处理，存在简化替代方案但外推能力下降
5. **评估规模有限**: 最大实验规模为 7B，在更大模型（70B+）上的表现尚未验证

### 潜在改进方向
1. 探索无需 landmark token 的简化实现，同时保持外推能力
2. 结合 [[Speculative Decoding|投机解码]] 进一步优化推理效率
3. 扩展到更大模型规模（70B+）验证泛化性
4. 自适应块大小和 top-K 数量，根据输入动态调整

### 可复现性评估
- [x] 代码开源（GitHub: Tencent-Hunyuan/HiLS-Attention）
- [ ] 预训练模型（未在论文中明确声明发布）
- [x] 训练细节完整（详细超参数见附录）
- [x] 数据集可获取（标准公开数据集）

---

## 关联笔记

### 基于
- [[Native Sparse Attention|NSA]]: 代表硬件对齐的稀疏注意力 kernel 设计
- [[Landmark Token]]: 用于块摘要的特殊 token
- [[HoPE]]: 混合位置编码，HiLS 的推荐位置编码方案

### 对比
- [[DashAttention]]: 可微分自适应稀疏层次注意力
- [[InfLLM]]: 密集-稀疏可切换注意力
- [[HSA-UltraLong]]: 层次化稀疏注意力
- [[RingAttention]]: 块状 Transformer 的近无限上下文方法

### 方法相关
- [[Chunk-wise Sparse Attention]]: 块级稀疏注意力的总体范式
- [[Hierarchical Softmax]]: HiLS 的核心分解机制
- [[Block Sparse Attention|Naive BSA]]: 块稀疏注意力的理论基准

### 硬件/数据相关
- [[Tensor Core]]: GPU 硬件加速单元，HiLS kernel 设计的优化目标
- [[GQA|Grouped-Query Attention]]: 多查询头共享 KV 头的注意力变体

---

## 速查卡片

> [!summary] Hierarchical Sparse Attention Done Right
> - **核心**: 通过层次化 softmax 分解实现端到端可学习的块稀疏注意力
> - **方法**: Landmark token 块摘要 + 线性质量代理 + 层次化因子分解
> - **结果**: 域内媲美全注意力，512 倍训练长度外推，90%+ 检索准确率
> - **代码**: https://github.com/Tencent-Hunyuan/HiLS-Attention

---

*笔记创建时间: 2026-07-10T16:30:00+08:00*
