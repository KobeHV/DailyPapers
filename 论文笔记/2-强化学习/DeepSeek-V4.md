---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
method_name: "DeepSeek-V4"
authors: [DeepSeek-AI]
year: 2026
venue: arXiv
tags: [mixture-of-experts, efficient-attention, long-context, reinforcement-learning, language-model, training-optimization]
zotero_collection: RL
image_source: local
arxiv_html: https://arxiv.org/html/2606.19348
created: 2026-07-16
---

# 论文笔记：DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | DeepSeek-AI |
| 日期 | April 2026 |
| 项目主页 | https://huggingface.co/collections/deepseek-ai/deepseek-v4 |
| 对比基线 | [[DeepSeek-V3]]、GPT-5.4、Gemini-3.1-Pro、Claude Opus 4.6、Kimi K2.6、GLM-5.1 |
| 链接 | [arXiv](https://arxiv.org/abs/2606.19348) / [Model](https://huggingface.co/collections/deepseek-ai/deepseek-v4) |

---

## 一句话总结

> DeepSeek-V4 通过混合注意力（CSA+HCA）、流形约束超连接（mHC）和 Muon 优化器三大创新，实现百万 token 上下文的高效处理，Pro-Max 模式在开源模型中达到新 SOTA。

---

## 核心贡献

1. **混合注意力架构（CSA + HCA）**: 设计 Compressed Sparse Attention 和 Heavily Compressed Attention 两种高效注意力机制并交错使用，将 1M-token 上下文的推理 FLOPs 降至 DeepSeek-V3.2 的 27%，KV Cache 降至 10%。
2. **Manifold-Constrained Hyper-Connections (mHC)**: 将残差映射约束到 Birkhoff 多面体，在保持模型表达力的同时显著增强深层训练的数值稳定性。
3. **Muon 优化器**: 首次在万亿参数级别的大规模训练中应用 Muon，配合 Hybrid Newton-Schulz 迭代实现更快收敛和更好的训练稳定性。
4. **双模型策略**: 发布 Pro（1.6T/49B 激活）和 Flash（284B/13B 激活）两个版本，分别针对最强性能和最高效率。
5. **On-Policy Distillation 统一**: 后训练阶段完全使用 OPD 替代传统混合 RL 阶段，将多个领域专家能力融合到单一统一模型。

---

## 问题背景

### 要解决的问题

推理模型（如 [[DeepSeek-R1]]）的 [[Test-Time Scaling]] 范式正受到 vanilla attention 二次复杂度的制约。同时，长周期任务（agentic workflows、跨文档分析等）对超长上下文的高效支持提出了迫切需求。如何打破超长上下文的效率瓶颈，使百万 token 上下文在实践上可行，是本文的核心目标。

### 现有方法的局限

- 标准 [[Multi-Head Attention|Transformer attention]] 在长序列下 FLOPs 和 KV Cache 均呈二次增长
- 现有开源模型在 1M-token 上下文场景下效率低下，无法在实际部署中经济地运行
- 稀疏注意力方法（如 sliding window）只能有限缓解问题

### 本文的动机

通过同时压缩 KV Cache 和稀疏化注意力计算，在不损伤建模能力的前提下实现数量级的效率提升。利用 [[Mixture-of-Experts|MoE]] 架构的效率优势，结合新型优化器加速收敛，构建真正能高效处理百万 token 的下一代大模型。

---

## 方法详解

### 模型架构

DeepSeek-V4 采用 **MoE Transformer + 混合注意力** 架构，保留 [[DeepSeekMoE]] 框架和 [[Multi-Token Prediction|MTP]] 策略，引入三大核心升级：

- **输入**: token 序列嵌入
- **Backbone**: 61 层 (Pro) / 43 层 (Flash) [[Transformer]]
- **注意力**: CSA 与 HCA 交错排列 + Sliding Window Attention 分支
- **FFN**: [[DeepSeekMoE]]（细粒度路由专家 + 共享专家）+ 前 3 层使用 [[Hash Routing]]
- **残差连接**: [[Manifold-Constrained Hyper-Connections|mHC]] 替代标准残差连接
- **优化器**: [[Muon Optimizer|Muon]]（多数参数）+ [[AdamW]]（嵌入/RMSNorm）
- **总参数**: Pro: 1.6T (49B 激活) / Flash: 284B (13B 激活)
- **上下文长度**: 1M tokens 原生支持

### 核心模块

#### 模块 1: 混合注意力 (Hybrid CSA + HCA)

**设计动机**: 在超长上下文下，注意力成为计算绝对瓶颈。通过 KV 压缩 + 稀疏注意力的组合，大幅降低 FLOPs 和 KV Cache 大小。

**CSA（压缩稀疏注意力）**:
- KV 压缩：每 $l_c=4$ 个 KV entries 压缩为 1 个（重叠压缩，实际压缩比 $1/l_c$）
- Lightning Indexer：低秩索引器快速计算 query-key 相关度评分
- Top-k 稀疏选择：每个 query 只关注 $k_s=512$ (Flash) / 1024 (Pro) 个压缩 KV entries
- Shared KV [[Multi-Query Attention|MQA]]：所有 head 共享一组压缩后的 KV
- 分组输出投影：减少最终输出投影的参数量

**HCA（重度压缩注意力）**:
- KV 压缩：每 $l_h=128$ 个 KV entries 压缩为 1 个（非重叠压缩）
- 保持稠密注意力（不稀疏选择）
- 同样使用 Shared KV MQA 和分组输出投影

**两者共用技术**:
- Partial [[Rotary Positional Embedding|RoPE]]（最后 64 维）+ core attention 输出位置抵消
- Sliding Window Attention 分支（窗口大小 $w=128$）：增强局部依赖建模
- [[Attention Sink]]：可学习 sink logit 允许 head"选择不关注"
- Query/KV RMSNorm：防止 attention logits 爆炸
- 混合精度存储：RoPE 维度 BF16，其余维度 FP8（KV cache 减少近半）

#### 模块 2: Manifold-Constrained Hyper-Connections (mHC)

**设计动机**: 标准 Hyper-Connections (HC) 虽然扩展了残差流的宽度维度，但在深层堆叠时频繁出现数值不稳定。mHC 通过流形约束解决此问题。

**核心设计**:
- 残差映射矩阵 $\mathbf{P}$ 约束到 [[Birkhoff Polytope|双随机矩阵流形]] $\mathcal{M}$ 上
- $\|\mathbf{P}\|_2 \leq 1$ 保证非扩张性，$\mathcal{M}$ 在矩阵乘法下封闭
- 投影通过 [[Sinkhorn-Knopp Algorithm]] 实现（迭代 20 次）
- 输入/输出映射通过 Sigmoid 约束为非负有界
- 动态参数化：参数 = 静态 bias + 动态（输入依赖）分量
- 扩展因子 $h_c=4$

#### 模块 3: Muon 优化器

**设计动机**: 相比 AdamW，Muon 通过对梯度矩阵进行正交化实现更快收敛和更好稳定性。

**[[Muon Optimizer]] 配置**:
- 嵌入层、预测头、RMSNorm 权重使用 AdamW（$\beta_1=0.9, \beta_2=0.95$）
- 其余模块使用 Muon（momentum=0.95, weight_decay=0.1）
- Hybrid [[Newton-Schulz Iterations]]：前 8 步 $(3.4445, -4.7750, 2.0315)$ + 后 2 步 $(2, -1.5, 0.5)$
- Nesterov 技巧 + update RMS rescale
- 不使用 QK-Clip（因为 attention 中有 RMSNorm）

---

## 关键公式

### 公式 1: [[Manifold-Constrained Hyper-Connections|mHC 残差状态更新]]

$$
\mathbf{x}_{t+1} = \mathbf{B}_t \mathbf{x}_t + \mathbf{A}_t \mathcal{F}(\mathbf{W}_t \mathbf{x}_t)
$$

**含义**: 第 $t$ 层残差状态更新，其中 $\mathbf{B}_t$ 为残差变换矩阵（约束到 Birkhoff 多面体），$\mathbf{A}_t$ 为输入映射，$\mathcal{F}$ 为实际层计算。

**符号说明**:
- $\mathbf{x}_t \in \mathbb{R}^{h_c \times d}$: 扩展后的残差状态（$h_c=4$ 为扩展因子）
- $\mathbf{B}_t \in \mathcal{M} \subset \mathbb{R}^{h_c \times h_c}$: 双随机残差映射矩阵
- $\mathbf{A}_t \in \mathbb{R}^{1 \times h_c}$: 输入映射（Sigmoid 约束）
- $\mathcal{F}$: 实际层计算（MoE 层）

### 公式 2: [[Birkhoff Polytope|Birkhoff 多面体约束]]

$$
\mathcal{M} = \{ \mathbf{P} \in \mathbb{R}_{\geq 0}^{n \times n} \mid \mathbf{P}\mathbf{1} = \mathbf{1}, \mathbf{1}^\top \mathbf{P} = \mathbf{1} \}
$$

**含义**: 约束残差映射矩阵为双随机矩阵，保证 $\|\mathbf{P}\|_2 \leq 1$ 且乘法封闭。

**符号说明**:
- $\mathbf{P}$: 双随机矩阵（行和=1，列和=1，非负）
- $n = h_c$: 扩展因子

### 公式 3: [[Sinkhorn-Knopp Algorithm|Sinkhorn-Knopp 投影]]

$$
\mathbf{P}^{(t)} = \mathcal{T}_c(\mathcal{T}_r(\mathbf{P}^{(t-1)}))
$$

其中 $\mathbf{P}^{(0)} = \exp(\tilde{\mathbf{P}})$，$\mathcal{T}_r, \mathcal{T}_c$ 分别为行和列归一化操作。

**含义**: 将非约束矩阵 $\tilde{\mathbf{P}}$ 投影到 Birkhoff 多面体上的迭代算法，$t_{\max}=20$。

### 公式 4: [[Compressed Sparse Attention|CSA 压缩 KV Entry 计算]]

$$
\mathbf{k}^{\text{Comp}}_j = \sum_{i} \alpha_i \mathbf{k}_i + \sum_{i} \beta_i \mathbf{v}_i
$$

$$
\alpha, \beta = \text{Softmax}_{\text{row}}\left([\mathbf{W}_\alpha \mathbf{x}_i + \mathbf{b}_\alpha; \mathbf{W}_\beta \mathbf{x}_i + \mathbf{b}_\beta]\right)
$$

**含义**: CSA 将每 $l_c$ 个 KV entries 压缩为一个，其中 $k$ 和 $v$ entries 各贡献一部分，通过可学习的压缩权重 $\alpha, \beta$ 加权组合。

**符号说明**:
- $\mathbf{k}_i, \mathbf{v}_i$: 原始 key 和 value entries
- $l_c = 4$: CSA 压缩率
- $\alpha_i, \beta_i$: 压缩权重（经由 Softmax 归一化）

### 公式 5: [[Compressed Sparse Attention|CSA Lightning Indexer 评分]]

$$
s_{t,j} = \sum_{m=1}^{n_h} w_m \cdot \text{ReLU}\left( \mathbf{q}_{I,m} \cdot \mathbf{k}^{\text{IComp}}_j \right)
$$

**含义**: query token $t$ 对压缩块 $j$ 的相关度评分，通过多个索引器 head 的加权和计算。

**符号说明**:
- $\mathbf{q}_{I,m}$: 低秩生成的索引器 query（$m=1,\ldots,n_h=64$）
- $\mathbf{k}^{\text{IComp}}_j$: 压缩后的索引器 key
- $w_m$: 可学习的 head 权重

### 公式 6: [[Heavily Compressed Attention|HCA 压缩 KV Entry]]

$$
\mathbf{k}^{\text{Comp}}_j = \sum_{i=(j-1)l_h}^{jl_h-1} \gamma_i \mathbf{k}_i
$$

$$
\gamma = \text{Softmax}(\mathbf{W}_\gamma \mathbf{x} + \mathbf{b}_\gamma)
$$

**含义**: HCA 以更大压缩比 $l_h=128$ 压缩 KV entries（非重叠），压缩权重仅来自 key entries。

### 公式 7: [[Attention Sink|Attention Sink 修正]]

$$
\alpha_{m,t,i} = \frac{\exp(s_{m,t,i})}{\exp(s_{m,t,i}) + \exp(\sigma_m)}
$$

**含义**: 通过可学习 sink logit $\sigma_m$ 允许每个 attention head 的总 attention score 可以不等于 1（甚至接近 0），减少噪声 token 干扰。

**符号说明**:
- $s_{m,t,i}$: 第 $m$ 个 head 在位置 $t$ 对 token $i$ 的 attention logit
- $\sigma_m$: 可学习的 sink logit

### 公式 8: [[Newton-Schulz Iterations|Hybrid Newton-Schulz 迭代]]

$$
\mathbf{X}_k = a \mathbf{X}_{k-1} + b (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top) \mathbf{X}_{k-1} + c (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1}
$$

**含义**: Muon 优化器中对梯度矩阵进行正交化的迭代算法，前 8 步使用 $(3.4445, -4.7750, 2.0315)$ 快速收敛，后 2 步使用 $(2, -1.5, 0.5)$ 精确稳定。

### 公式 9: [[On-Policy Distillation|OPD 目标函数]]

$$
\mathcal{L}_{\text{OPD}}(\theta) = \sum_{k=1}^{K} \lambda_k \cdot D_{KL}(\pi_\theta \| \pi_k)
$$

**含义**: 统一模型 $\pi_\theta$ 在自身生成轨迹上学习多个教师专家 $\{\pi_k\}_{k=1}^K$ 的输出分布，通过 reverse KL 散度实现知识融合。

**符号说明**:
- $\lambda_k$: 专家权重（按领域重要性分配）
- $\pi_k$: 第 $k$ 个领域专家模型（数学/编程/Agent/指令遵循等）
- $K > 10$: 教师模型数量

---

## 关键图表

### Figure 1: Benchmark Performance and Efficiency Comparison

![[deepseek_v4_p1_i0.png]]
![[deepseek_v4_p1_i1.png]]

**说明**: **左图**: DeepSeek-V4-Pro-Max 与 Claude Opus 4.6 Max、GPT-5.4 xHigh、Gemini-3.1-Pro-High 在知识/推理/Agent benchmark 上的性能对比。V4-Pro-Max 在开源模型中达到 SOTA。**右图**: DeepSeek-V4 系列与 V3.2 在 1M-token 上下文下的单 token 推理 FLOPs 和累积 KV Cache 对比，V4-Pro 仅需 27% FLOPs 和 10% KV Cache。

### Figure 2: Overall Architecture

![[deepseek_v4_page6.png]]

**说明**: DeepSeek-V4 整体架构。使用 CSA/HCA 混合注意力、DeepSeekMoE FFN、mHC 强化残差连接。MTP 模块和预测头保持不变。

### Figure 3: CSA Core Architecture

![[deepseek_v4_page9.png]]

**说明**: [[Compressed Sparse Attention|CSA]] 的核心架构。先通过 Token-Level Compressor 压缩 KV entries 至 $1/l_c$，再用 Lightning Indexer 进行 top-k 稀疏选择，结合 sliding window KV entries 进行 Shared KV MQA 注意力计算。

### Figure 4: HCA Core Architecture

![[deepseek_v4_page11.png]]

**说明**: [[Heavily Compressed Attention|HCA]] 的核心架构。以更大压缩比 $l_h$ 压缩 KV entries，保持稠密注意力，同样结合 sliding window 和 Shared KV MQA。

### Figure 5: Fine-Grained EP Scheme

![[deepseek_v4_page15.png]]

**说明**: [[Expert Parallelism|细粒度专家并行]]方案对比。(a) Naive: 顺序执行。(b) Comet: Dispatch 与 Linear-1 重叠，Linear-2 与 Combine 重叠。(c) Ours: 将专家分批调度为 waves，实现通信与计算的持续流水线重叠，理论 speedup 1.92x。

### Figure 6: KV Cache Layout

![[deepseek_v4_page22.png]]

**说明**: DeepSeek-V4 的异构 KV Cache 管理方案。分为 **State Cache**（SWA 和未完成压缩的 tail tokens）和 **Classical KV Cache**（CSA/HCA 压缩后的 KV entries）。每个 cache block 覆盖 $\text{lcm}(l_c, l_h)$ 个原始 token。

### Figure 7: Interleaved Thinking Management

![[deepseek_v4_page31.png]]

**说明**: (a) Tool-calling 场景：保留完整推理历史跨所有轮次（利用 1M 上下文窗口）。(b) 通用对话场景：新用户消息到达时丢弃历史推理内容（保持上下文简洁）。

### Figure 8: Formal Reasoning Results

![[deepseek_v4_page40.png]]

**说明**: **左**: Practical Regime（Putnam-200 Pass@8）：DeepSeek-V4-Flash-Max 以 81.0 远超 Seed-2.0-Pro (35.5)。**右**: Frontier Regime（Putnam-2025）：DeepSeek-V4 在混合 formal-informal 推理 + 大规模 compute 下达到满分 120/120。

### Figure 9: MRCR Performance

![[deepseek_v4_page40.png]]

**说明**: DeepSeek-V4 系列在 MRCR (Multi-needle Retrieval) 任务上的表现。128K 上下文中检索性能高度稳定，1M token 下仍保持强劲检索能力。

### Figure 10: HLE and TerminalBench by Reasoning Effort

![[deepseek_v4_page41.png]]

**说明**: **左**: HLE 上不同推理模式（Non-think, Think High, Max）的性能和 token 效率对比。V4-Pro 相比 V3.2 展现更高 token 效率。**右**: TerminalBench 2.0 上的类似对比。

### Figure 11 & 12: White-Collar Task Evaluation

![[deepseek_v4_page43.png]]

**说明**: **Fig 11**: V4-Pro-Max vs Opus-4.6-Max 在分析/生成/编辑三类白领任务上的胜率对比，总体 Win 53% vs Lose 37%。**Fig 12**: 四维度评分对比，V4-Pro-Max 在任务完成度和内容质量上优势显著。

### Figure 13-15: Example Outputs

![[deepseek_v4_page56.png]]

**说明**: 真实任务案例：(13) 奶茶品牌+北京地铁联合营销方案，(14) NASDAQ 定投策略对比分析，(15) 2020-2025 诺贝尔科学奖研究报告。

---

## 关键表格

### Table 1: Base Model Comparison

| Benchmark | DeepSeek-V3.2-Base | DeepSeek-V4-Flash-Base | DeepSeek-V4-Pro-Base |
|-----------|-------------------|----------------------|---------------------|
| # Activated Params | 37B | **13B** | 49B |
| # Total Params | 671B | **284B** | 1.6T |
| MMLU (5-shot) | 87.8 | 88.7 | **90.1** |
| MMLU-Pro (5-shot) | 65.5 | 68.3 | **73.5** |
| SimpleQA-Verified (25-shot) | 28.3 | 30.1 | **55.2** |
| HumanEval (0-shot) | 62.8 | 69.5 | **76.8** |
| MATH (4-shot) | 60.5 | 57.4 | **64.5** |
| LongBench-V2 (1-shot) | 40.2 | 44.7 | **51.5** |

**说明**: V4-Flash-Base 以更少的激活/总参数量超越 V3.2-Base，体现了架构和训练优化的收益。V4-Pro-Base 在所有维度全面领先。

### Table 2 & 3: Reasoning Modes

| Mode | 特点 | 典型场景 |
|------|------|----------|
| Non-think | 快速直觉响应 | 日常任务、紧急反应 |
| Think High | 有意识逻辑分析 | 复杂问题求解、中等风险决策 |
| Think Max | 推理到极限 | 探索模型推理能力边界 |

**Think Max 系统提示**指示模型"完全彻底地思考，对逻辑进行压力测试，记录所有中间步骤和备选方案"。

### Table 4: Tool-Call Schema

DeepSeek-V4 引入基于 `|DSML|` 特殊 token 的 XML 格式工具调用模式：

```
<|DSML|tool_calls>
<|DSML|invoke name="TOOL_NAME">
<|DSML|parameter name="PARAM" string="true">VALUE</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>
```

相比 JSON 格式，XML 有效减少了转义错误。

### Table 5: Quick Instruction Special Tokens

| Token | 功能 |
|-------|------|
| `<|action|>` | 判断是否需要搜索 |
| `<|title|>` | 生成对话标题 |
| `<|query|>` | 生成搜索查询 |
| `<|authority|>` | 判断来源权威性需求 |
| `<|domain|>` | 识别用户 prompt 领域 |
| `<|extracted_url|>` `<|read_url|>` | URL 提取和判断是否阅读 |

复用已有 KV Cache，避免重复 prefill，降低 Time-to-First-Token。

### Table 6: Main Benchmark Results (V4-Pro-Max vs Frontier Models)

DeepSeek-V4-Pro-Max 在主要 benchmark 上与前沿闭源/开源模型对齐：
- **HLE (Pass@1)**: 90.1 (vs GPT-5.4: 39.8, Gemini-3.1-Pro: 44.4, K2.6: 90.5)
- **Codeforces (Rating)**: 3206（全球排名第 23 位）
- **Apex Shortlist**: 90.2 (vs GPT-5.4: 78.1)
- **MRCR 1M**: 83.5 (vs Opus-4.6: 92.9, Gemini-3.1-Pro: 76.3)

### Table 7: Reasoning Mode Ablation

| Mode | V4-Flash HLE | V4-Pro HLE | V4-Flash Codeforces | V4-Pro Codeforces |
|------|-------------|-----------|--------------------|--------------------|
| Non-Think | 8.1 | 7.7 | - | - |
| Think High | 29.4 | 34.5 | 2816 | 2919 |
| Think Max | 34.8 | 37.7 | 3052 | 3206 |

**说明**: Max 模式使用更长上下文 + 降低 RL 长度惩罚，在最具挑战性的任务上显著优于 High 模式。

### Table 8: R&D Coding Benchmark

| Model | Haiku 4.5 | Sonnet 4.5 | V4-Pro-Max | Opus 4.5 | Opus 4.5 Thinking | Opus 4.6 Thinking |
|-------|----------|-----------|-----------|---------|-------------------|-------------------|
| Pass Rate | 13% | 47% | 67% | 70% | 73% | 80% |

内部开发者调查（n=85）：52% 认为 V4-Pro 可作为主力编程模型，39% 倾向同意，<9% 反对。

---

## 训练细节

### 预训练数据

32T+ tokens，包含数学、代码、网页、长文档、多语言等高多样性高质量语料。特别强调长文档数据（科学论文、技术报告等）。继承 V3 的 tokenizer（128K 词汇量）、FIM 策略和文档打包策略。

### 预训练设置

| 配置 | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
|------|-------------------|-----------------|
| 层数 | 43 | 61 |
| 隐藏维度 | 4096 | 7168 |
| 训练 Token | 32T | 33T |
| 最大 Batch Size | 75.5M | 94.4M |
| 峰值学习率 | $2.7 \times 10^{-4}$ | $2.0 \times 10^{-4}$ |
| 序列长度调度 | 4K $\to$ 16K $\to$ 64K $\to$ 1M | 同 Flash |
| 稀疏注意力引入 | 前 1T dense warmup | 更长 dense warmup |
| MTP 深度 | 1 | 1 |

### 训练稳定性

- **[[Anticipatory Routing]]**: 路由网络使用历史参数，打破路由-异常值反馈循环，动态启用（约 20% 额外 wall-time 开销）
- **[[SwiGLU Clamping]]**: 线性分量 [-10, 10]，门控分量上限 10
- 二者组合有效解决万亿参数 MoE 训练中的 loss spike

### 后训练流程

1. **专家训练 (Specialist Training)**: 对数学、编程、Agent、指令遵循等领域分别训练，SFT + [[Group Relative Policy Optimization|GRPO]] RL

2. **[[Generative Reward Model|生成式奖励模型 (GRM)]]**: 替代传统标量 RM，模型原生充当 GRM，仅需少量人工标注

3. **[[On-Policy Distillation|On-Policy Distillation (OPD)]]**: 使用超过 10 个教师模型，通过 full-vocabulary reverse KL 损失融合到统一 student 模型
   - 缓存 teacher 最后层 hidden states 避免完整 logits 存储
   - 按 teacher index 排序训练样本，每次只加载一个 teacher head

4. **[[FP4 Quantization-Aware Training|FP4 QAT]]**: 对 MoE 专家权重 + CSA Indexer QK 路径进行 FP4 量化感知训练，FP4-to-FP8 去量化为无损操作

---

## 基础设施优化

### 训练框架

- **[[Expert Parallelism|细粒度 EP]]**: 通信计算 overlap 达 1.92x speedup，开源自研 MegaMoE 大核
- **[[TileLang]] DSL**: 融合数百个 Atten 算子，平衡开发效率与运行时性能
- **[[DeepGEMM]]**: 替代 cuBLAS，batch-invariant + deterministic 矩阵乘法
- **[[DualPipe]]**: 双流水线调度，mHC 额外 wall-time 仅 6.7%
- **[[Tensor-Level Activation Checkpointing]]**: 基于 TorchFX 的细粒度重计算控制

### 推理框架

- **异构 KV Cache 管理**: State Cache（SWA + 尾 token）+ Classical KV Cache（压缩 entries）
- **On-Disk KV Cache 存储**: CSA/HCA 压缩 entries 直接存盘；SWA entries 三种策略（Full/Periodic/Zero Caching）
- **预取 + 容错**: Token 粒度的 Write-Ahead Log，支持抢占和故障恢复

---

## 批判性思考

### 优点

1. **效率突破显著**: 1M-token 上下文下 FLOPs 降至 27%、KV Cache 降至 10%，使百万 token 上下文在实践中可行
2. **开源 SOTA**: V4-Pro-Max 在大多数基准上超越所有开源模型，在 Codeforces 上与 GPT-5.4 相当
3. **双模型策略合理**: Pro 追求极致性能，Flash 追求极致性价比
4. **训练细节极其透明**: 完整披露架构、优化器、训练稳定性方案、基础设施优化，对社区有极高参考价值
5. **工程与算法并重**: 从 TileLang DSL 到 MegaMoE 大核到 FP4 QAT，系统性优化了整个技术栈

### 局限性

1. **架构复杂度高**: 作者自承"为了降低风险保留了许多已验证的组件和技巧，使架构相对复杂"，CSA/HCA/mHC/MoE/MTP 等多种技术叠加
2. **训练稳定性机理解释不足**: Anticipatory Routing 和 SwiGLU Clamping 为什么有效"仍然是一个悬而未决的问题"
3. **与前沿闭源模型仍有差距**: 在知识基准（MMLU-Pro、SimpleQA）上仍落后于 Gemini-3.1-Pro
4. **小模型知识能力弱**: V4-Flash 由于参数规模较小，在知识密集型任务上表现明显不如 Pro 版本

### 潜在改进方向

1. **架构简化**: 通过更系统的消融实验精简组件，找到 minimal yet effective 的设计
2. **训练稳定性理论**: 深入理解 MoE 训练不稳定的根本原因，走向可预测的稳定训练
3. **多模态扩展**: 论文明确提到正在将多模态能力融入模型
4. **在线学习**: 1M 上下文为在线学习等新范式提供了基础
5. **新稀疏维度**: 探索更稀疏的嵌入模块等新稀疏性方向

### 可复现性评估

- [x] 代码开源（推理代码 + MegaMoE 内核）
- [x] 预训练模型（HuggingFace）
- [x] 训练细节完整
- [ ] 训练数据不可获取
- [ ] 完整训练代码未开源

---

## 关联笔记

### 基于
- [[DeepSeek-V3]]: 继承 DeepSeekMoE、MTP、DualPipe、Auxiliary-Loss-Free 等核心设计
- [[DeepSeek-R1]]: 后训练 GRPO 策略
- [[DeepSeek-V2]]: DeepSeekMoE 框架

### 对比
- GPT-5.4: 闭源最强推理模型之一
- Gemini-3.1-Pro: 知识基准最强闭源模型
- Claude Opus 4.6: 长上下文和创意写作的有力竞争者
- Kimi K2.6: 开源 Agent 最强竞争者
- GLM-5.1: 开源模型有力竞争者

### 方法相关
- [[Compressed Sparse Attention]]: 核心注意力机制
- [[Heavily Compressed Attention]]: 核心注意力机制
- [[Manifold-Constrained Hyper-Connections]]: 残差连接升级
- [[Muon Optimizer]]: 核心优化器
- [[On-Policy Distillation]]: 后训练统一方法
- [[Group Relative Policy Optimization]]: 后训练 RL 算法
- [[Generative Reward Model]]: 替代标量 RM

### 硬件/数据相关
- [[Expert Parallelism]]: 细粒度通信计算重叠
- [[TileLang]]: 内核开发 DSL
- [[DeepGEMM]]: 矩阵乘法内核库
- [[FP4 Quantization-Aware Training]]: 推理加速
- [[Tensor-Level Activation Checkpointing]]: 显存优化
- [[DualPipe]]: 流水线并行调度

---

## 速查卡片

> [!summary] DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
> - **核心**: 通过 CSA+HCA 混合注意力、mHC 和 Muon 实现百万 token 上下文高效处理
> - **方法**: 混合注意力压缩 + 稀疏选择 + 双随机矩阵约束残差 + Newton-Schulz 优化
> - **结果**: Pro-Max 达到开源 SOTA，1M 上下文 FLOPs 降至 V3.2 的 27%，Codeforces 3206 分
> - **代码**: https://huggingface.co/collections/deepseek-ai/deepseek-v4

---

*笔记创建时间: 2026-07-16*
