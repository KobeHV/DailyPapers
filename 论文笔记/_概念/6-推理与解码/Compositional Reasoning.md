---
tags: [reasoning, composition, llm, cognitive-science]
aliases: [组合推理, Compositional Generalization]
created: 2026-07-09
---

# Compositional Reasoning

## 定义

Compositional Reasoning（组合推理）指将多个原始技能（primitives）系统地组合成新的、更高层次的推理策略的能力。这是人类智能的标志性特征之一，也是评估 AI 系统泛化能力的关键维度。

## 核心概念

### Primitives（原始技能）
不能再分解的基本操作单元。例如：
- 数学中：加法、乘法、因式分解等基本运算。
- 符号推理中：单个重写规则 $A \to B$。
- 编程中：基本语法转换、变量替换。

### Composition Types（组合类型）

#### Sequential Composition（顺序组合）
将多个有依赖关系的操作按顺序合并为一个操作：
$$r_1 \circ r_2(s) = t \quad \text{即将 } s \xrightarrow{r_1} s' \xrightarrow{r_2} t \text{ 压缩为 } s \xrightarrow{r_1 \circ r_2} t$$

#### Parallel Composition（并行组合）
将多个作用于不同子项的独立操作合并为一个操作：
$$r_a \parallel r_b \quad \text{当 } r_a \text{ 和 } r_b \text{ 作用于不相交的子项时}$$

#### Hierarchical Composition（层次组合）
组合后的操作本身可以进一步被组合，形成多层嵌套结构。

## 关键问题

### 1. Amplification vs Composition
后训练到底是在放大已有模式，还是创造了新的组合策略？

- **Amplification 观点**: RL 仅重新加权已有的推理模式（从 ~20% 提升到 ~90%）。
- **Composition 观点**: RL 能够发现并巩固训练分布中不存在的新组合策略。

[[RL Post-Training]] 提供了支持 composition 观点的因果证据。

### 2. Compositional Generalization
模型能否在未见过的组合上表现良好？

关键维度：
- **Systematicity**: 理解并应用组合规则的能力。
- **Productivity**: 从有限原始技能生成无限新组合的能力。
- **Length Generalization**: 处理比训练样本更长的推理链的能力。

### 3. Gating Factors（组合能力出现的门槛条件）
根据 [[RL Post-Training]] 的预训练消融实验：
- **必要条件**: 预训练必须将原始技能组织为**过程性程序** (procedural reduction procedures)。
- **非充分条件**: 仅接触原始技能（无程序结构）不足以支持 RL 后的组合涌现。

## 评估方法

### 可控环境
- **改写语法 (Rewrite Grammar)**: 定义形式化的重写规则，每条 trace 可审计。
- **合成任务**: 控制训练分布，构造需要组合才能解决的留出问题。

### Trace-Level Auditing（轨迹级审计）
对每一步推理进行分类：
- 原始（primitive）
- 顺序组合（sequential composition）
- 并行组合（parallel composition）
- 无效捷径（invalid shortcut）

## 与 RL Post-Training 的关系

[[RL Post-Training]] 的核心发现：
1. RL 在 **Phase 1** 先加强原始技能。
2. RL 在 **Phase 2** 自发发现顺序和并行组合。
3. 组合后的宏规则被**复用和巩固**成稳定的 repertoire。
4. 选择性（selectivity）是 RL 区别于 RFT 的关键。

## 开放问题

1. 组合能力是否随模型规模呈涌现性增长（emergent at scale）？
2. 不同组合结构（线性链 vs 树 vs 图）的可学习性差异？
3. 如何设计预训练数据以最大化后训练阶段的组合能力？
4. 组合推理在真实 LLM 中是否遵循相同的阶段性动力学？
5. 过程奖励模型（PRM）能否加速组合发现？

## 相关笔记

- [[RL Post-Training]]: 核心论文，通过可控实验证明 RL 能组合原始技能
- [[GRPO]]: 实现组合推理的 RL 算法
- [[DeepSeek-R1]]: 真实世界中的 RL 后训练组合推理案例
- [[From f(x) and g(x) to f(g(x))]]: 互补证据，LLM 在 RL 中通过组合已有技能获得新能力
- [[How Does RL Post-Training Induce Skill Composition?]]: 树形结构的组合泛化研究
