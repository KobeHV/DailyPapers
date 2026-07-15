---
title: "TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs"
method_name: "TreeThink"
authors: [Burak S. Akbudak, Zeynel A. Uluşan, Can S. Erer, Gözde Gül Şahin]
year: 2026
venue: arXiv
tags: [tree-search, llm-reasoning, mathematical-reasoning, formal-proof, mcts, open-source, neural-theorem-proving]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2607.11258v1
created: 2026-07-15
---

# 论文笔记：TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Bogazici University, Friedrich-Alexander-Universität Erlangen-Nürnberg |
| 日期 | July 2026 |
| 项目主页 | https://github.com/GGLAB-KU/treethink |
| 链接 | [arXiv](https://arxiv.org/abs/2607.11258) / [Code](https://github.com/GGLAB-KU/treethink) / [PyPI](https://pypi.org/project/treethink/) |

---

## 一句话总结

> TreeThink 是一个模块化的开源 Python 库，将树搜索策略与 LLM 推理解耦，支持 MCTS、BFS、BeamSearch 等多种搜索算法与形式化证明验证的无缝集成。

---

## 核心贡献

1. **模块化四组件架构**: 将树搜索解耦为 Methods（搜索策略）、Policies（LLM 生成）、Evaluators（节点评分）、Termination（证明验证），每个组件可独立替换
2. **异步执行引擎**: 全异步树搜索实现，通过 `--async` 参数自动转换所有组件为异步等价物，实现最高 6.3x 的墙钟加速比
3. **形式化证明集成**: 原生支持 Lean 4（Kimina server）和 Rocq/Coq 8.20 的 REPL 实时验证，采用两阶段验证系统（遇到时验证 + 搜索结束后批量验证）

---

## 问题背景

### 要解决的问题
现有的 LLM 推理树搜索实现通常是紧耦合的：搜索逻辑、模型推理、评分函数交织在一起，难以复用、扩展和维护。同时，面向通用 LLM 推理的树搜索库缺乏形式化证明验证器的集成。

### 现有方法的局限
- **通用树搜索库**（如 Tree-of-Thoughts 实现）不支持形式化证明验证器的接入
- **定理证明系统**使用任务特定的搜索实现，代码复用性差
- **搜索、策略、评估、验证**四个环节紧耦合，替换任一组件都需要修改大量代码

### 本文的动机
通过将树搜索的四个核心环节解耦为可插拔组件，构建一个既支持通用自然语言推理、又支持形式化数学证明验证的模块化框架，降低研究者在不同搜索策略和评估方法间切换的成本。

---

## 方法详解

### 系统架构

TreeThink 采用 **四组件流水线** 架构：
- **输入**: 数学问题表述（自然语言或形式化语言）
- **核心循环**: 扩展 → 评分 → 选择 → 终止
- **Methods**: 定义搜索策略（如何选择下一个扩展节点）
- **Policies**: 管理 LLM 推理（如何生成子节点）
- **Evaluators**: 评估节点质量（如何给节点打分）
- **Termination**: 验证证明完整性（何时终止搜索）
- **输出**: 已验证的完整证明

### 核心组件

#### 组件 1: Methods — 搜索策略

**设计动机**: 利用 [[UCT|UCB1]] 等[[蒙特卡洛树搜索|MCTS]]选择策略，在探索与利用间取得平衡。

**支持的方法**:
- **[[蒙特卡洛树搜索|MCTS (Monte Carlo Tree Search)]]**: 使用 [[UCT|UCB1]] 选择策略，通过 `exploration_weight` 参数控制探索强度
- **[[广度优先树搜索|BFTS (Breadth-First Tree Search)]]**: 逐层扩展，适合浅层宽搜索
- **[[束搜索|BeamSearch]]**: 维护固定大小的 top-k 节点束，平衡搜索宽度与深度

所有方法均提供同步（sync）和异步（async）变体，通过 `--async` 标志自动转换。

#### 组件 2: Policies — LLM 策略（子节点生成）

**设计动机**: 利用 [[vLLM]] 提供的高效批推理能力，从当前节点生成候选证明步骤。

**支持的策略**:
- `vllm_policy`: 标准本地 [[vLLM]] 模型，支持 [[LoRA]] 适配器
- `dynamic_policy`: 支持动态调整采样参数（temperature、top-p 等）
- `vllm_server_policy`: 通过 OpenAI 兼容 API 连接外部 [[vLLM]] 服务器

生成过程：从根节点（问题陈述）开始，对每个待扩展节点调用 LLM 生成候选下一步证明步骤。

#### 组件 3: Evaluators — 节点评估

**设计动机**: 通过多维度评估引导搜索向最有希望的分支前进。

**支持的评估器**:
- `cumulative_logprob_evaluator`: 使用 LLM 生成路径的累积对数概率作为评分
- `repl_evaluator`: 通过形式化语言服务器的 [[REPL]] 验证提供二元反馈（通过/不通过）
- `llm_as_judge_evaluator`: 使用辅助 [[LLM]] 作为裁判评估证明质量
- `pairwise_tournament_evaluator`: 兄弟节点间的单败淘汰赛
- `normalized_lengths_evaluator`: [[BFS-Prover]] 风格评分（`logprob / L^alpha`），对长序列进行长度归一化
- `rocq_evaluator`: [[Rocq]] ([[Coq]]) 证明验证

#### 组件 4: Termination — 证明验证

**设计动机**: 通过两阶段验证系统确保搜索终止时返回的是完整、正确的证明。

**两阶段验证系统**:
1. **On-encounter（快速路径）**: 在搜索过程中立即验证遇到的终止节点
2. **On-paths（后备）**: 搜索完成后批量验证所有终止叶节点

**支持的形式化语言**:
- **[[Lean 4]]**: 通过 Kimina server 支持同步和异步验证
- **[[Rocq]] (Coq 8.20)**: 通过 `rocq-ml-server` 支持同步验证
- **[[Isabelle|Isabelle/HOL]]**: 待实现

---

## 关键公式

### 公式 1: [[UCT|UCB1 选择策略]]

$$
UCB1 = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}
$$

**含义**: MCTS 中用于平衡探索与利用的节点选择公式

**符号说明**:
- $w_i$: 节点 $i$ 的累计奖励
- $n_i$: 节点 $i$ 的访问次数
- $N$: 父节点的总访问次数
- $c$: 探索权重（exploration_weight），控制探索强度

### 公式 2: [[累积对数概率|Cumulative Log-Probability 评估器]]

$$
\text{score}(s) = \sum_{t=1}^{|s|} \log P(t_t | t_{<t}, c)
$$

**含义**: 对整个生成路径的累积对数概率求和，作为节点质量的评分

**符号说明**:
- $s$: 从根到当前节点的生成路径（token 序列）
- $t_t$: 路径中第 $t$ 个 token
- $t_{<t}$: 第 $t$ 个 token 之前的所有 token
- $c$: 上下文（问题陈述 + 已生成的证明步骤）
- $P(t_t | t_{<t}, c)$: LLM 在第 $t$ 步的条件概率

### 公式 3: [[归一化长度评分|Normalized Lengths 评估器（BFS-Prover 风格）]]

$$
\text{score}(s) = \frac{\sum_{t=1}^{|s|} \log P(t_t | t_{<t}, c)}{|s|^\alpha}
$$

**含义**: 对累积对数概率进行长度归一化，防止长序列获得不公平的低分

**符号说明**:
- $|s|$: 生成路径的 token 长度
- $\alpha$: 长度惩罚系数（BFS-Prover 风格）
- 其余符号与公式 2 相同

---

## 关键图表

### Figure 1: NTP 系统流程图 / TreeThink 架构概览

![Figure 1](https://arxiv.org/html/2607.11258v1/figures/ntp_process_v4_cropped.png)

**说明**: TreeThink 的整体架构图。展示从问题输入开始，经过扩展（Policies）、评分（Evaluators）、选择（Methods）、验证（Termination）的完整搜索循环，最终输出已验证证明。

### Table 1: TreeThink 组件分类

| 组件 | 功能 | 实现 |
|------|------|------|
| Methods | 搜索策略（如何探索树） | MCTS (UCB1), BFTS, BeamSearch |
| Policies | LLM 推理（如何生成子节点） | vllm_policy, dynamic_policy, vllm_server_policy |
| Evaluators | 节点评估（如何评分） | cumulative_logprob, repl, llm_as_judge, pairwise_tournament, normalized_lengths, rocq |
| Termination | 证明验证（何时终止） | On-encounter（快速路径）+ On-paths（批量后备） |

**说明**: TreeThink 的四组件架构。每个组件负责树搜索的一个独立环节，可独立替换和组合。

### Table 2: 异步执行与并发级别的影响

| 配置 | 墙钟时间 | 加速比 | 说明 |
|------|----------|--------|------|
| 同步 (sync) | baseline | 1.0x | 无并发 |
| 异步 (async) 低并发 | - | - | 有限并行度 |
| 异步 (async) 高并发 | - | **最高 6.3x** | 最大并行度 |

**说明**: 异步执行在不同并发级别下的加速效果，最高实现 6.3x 墙钟加速比。

### Table 3: MATH500 自然语言数学推理结果

| 方法 | 搜索算法 | 评估器 | MATH500 准确率 |
|------|----------|--------|----------------|
| Baseline (greedy) | - | - | baseline |
| TreeThink | MCTS | cumulative_logprob | - |
| TreeThink | BFTS | normalized_lengths | - |
| TreeThink | BeamSearch | llm_as_judge | - |

**说明**: TreeThink 在 MATH500 上的自然语言数学推理表现，不同搜索算法与评估器组合的结果对比。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[miniF2F]] | 488 题 | IMO/AIME/AMC 等数学竞赛题的形式化版本 | 形式化证明搜索评估 |
| [[MATH500]] | 500 题 | MATH 数据集的子集，高中竞赛数学 | 自然语言推理评估 |

### 实现细节

- **Backbone**: 支持多种 [[vLLM]] 兼容模型（如 InternLM2-7B）
- **推理引擎**: [[vLLM]]（本地或服务器模式）
- **形式化语言支持**: [[Lean 4]]（Kimina server）、[[Rocq]]/[[Coq]] 8.20（rocq-ml-server）
- **配置方式**: YAML/TOML 配置文件
- **执行模式**: 同步（sync）和异步（async，通过 `--async` 标志）

### 关键结果

1. **异步加速**: 异步执行最高实现 6.3x 墙钟加速比，说明树搜索中节点评估（尤其是 REPL 验证）是高度可并行的
2. **跨语言形式化搜索**: 支持 Lean 4 和 Rocq 两种形式化语言，展示了库的通用性
3. **自然语言推理**: 在 MATH500 上展示自然语言数学推理能力，验证了库的灵活性

---

## 批判性思考

### 优点
1. **高度模块化设计**: Methods/Policies/Evaluators/Termination 四组件架构清晰，各组件可独立替换，降低了新搜索策略和评估方法的实现成本
2. **异步优先**: 全异步执行设计充分利用了树搜索的天然并行性，6.3x 加速比验证了实际收益
3. **形式化与自然语言统一框架**: 同时支持形式化证明验证（Lean 4/Rocq）和自然语言推理，填补了通用树搜索库与定理证明系统之间的空白
4. **开源生态友好**: MIT 许可、vLLM 集成、PyPI 发布，降低了使用门槛

### 局限性
1. **评估实验不够全面**: 搜索结果未提供每个搜索算法的具体准确率数值（如 miniF2F pass@k），难以定量评估不同方法的优劣
2. **Isabelle/HOL 未实现**: 三大主流证明助手中 Isabelle/HOL 尚未支持，限制了跨语言比较的完整性
3. **搜索方法的原创性有限**: 库本身的贡献在于集成和工程化，MCTS、BFS、BeamSearch 均为已有算法
4. **未见与 SoTA 方法的对比**: 未与 DeepSeek-Prover、Goedel-Prover 等专用定理证明系统的搜索策略进行直接比较

### 潜在改进方向
1. 添加更多搜索策略（如 [[A*搜索]]、[[AlphaZero]] 风格的自对弈搜索）
2. 实现自适应搜索策略选择（根据问题难度动态切换搜索方法）
3. 支持 [[Isabelle|Isabelle/HOL]] 验证
4. 集成 [[DSPy]] 等自动化 prompt 优化框架
5. 提供可复现的基准实验和预训练模型检查点

### 可复现性评估
- [x] 代码开源（MIT 许可，GitHub + PyPI）
- [ ] 预训练模型（使用现成的 LLM，未提供专用模型）
- [x] 训练细节完整（配置驱动，YAML/TOML）
- [x] 数据集可获取（miniF2F 和 MATH500 均为公开基准）

---

## 关联笔记

### 基于
- [[蒙特卡洛树搜索|MCTS (Monte Carlo Tree Search)]]: 核心搜索算法之一，使用 UCB1 选择
- [[束搜索|BeamSearch]]: 经典搜索算法，TreeThink 的实现之一
- [[BFS-Prover]]: normalized lengths 评估器的来源
- [[Tree-of-Thought]]: 树搜索增强 LLM 推理的代表性工作

### 对比
- [[llm-reasoners]]: 另一个 LLM 推理库，TreeThink 更侧重形式化证明和模块化设计
- [[Tree-of-Thought|ToT]]: TreeThink 提供了更通用的框架，而 ToT 是特定搜索策略

### 方法相关
- [[UCT|UCB1]]: MCTS 中的节点选择策略
- [[vLLM]]: 核心推理引擎
- [[REPL]]: 实时证明验证的接口

### 硬件/数据相关
- [[miniF2F]]: 形式化证明基准
- [[MATH500]]: 自然语言推理基准

---

## 速查卡片

> [!summary] TreeThink: A Modular Tree Search Library for Mathematical Reasoning with LLMs
> - **核心**: 模块化树搜索库，解耦搜索/策略/评估/验证四组件
> - **方法**: MCTS(UCB1) / BFTS / BeamSearch + 多种评估器 + Lean 4/Rocq 验证
> - **结果**: 最高 6.3x 异步加速，支持 miniF2F 和 MATH500
> - **代码**: https://github.com/GGLAB-KU/treethink

---

*笔记创建时间: 2026-07-15*
