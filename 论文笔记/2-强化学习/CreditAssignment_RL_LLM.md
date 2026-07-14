---
title: "From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models"
method_name: "CreditAssignment_RL_LLM"
authors: [Chenchen Zhang]
year: 2026
venue: arXiv
tags: [survey, credit-assignment, reinforcement-learning, reasoning, agentic, llm, taxonomy]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2604.09459v1
created: 2026-07-14
updated: 2026-07-14
aliases: [Credit Assignment Survey, CA for LLM RL]
---

# 论文笔记：Credit Assignment in RL for LLMs — 综述

## 元信息

| 项目 | 内容 |
|------|------|
| **作者** | Chenchen Zhang (Independent Researcher) |
| **发表** | arXiv:2604.09459, 2026年4月 |
| **覆盖范围** | **47 种方法**（41 核心 + 6 辅助），发表于 2024 初 - 2026 初 |
| **链接** | [arXiv](https://arxiv.org/abs/2604.09459) |

---

## 一句话总结

> 覆盖 47 种 LLM RL 信用分配方法的系统综述，提出**二维分类体系**（粒度 × 方法论），核心洞察：Reasoning RL → Agentic RL 并非简单延伸而是**问题性质的根本转变**。

---

## 什么是信用分配？

信用分配问题：当模型在长轨迹结束时获得奖励，如何确定**哪些** token/步骤/turn 真正**导致**了成功或失败？

$$\text{奖励 } R \rightarrow \text{需要分配给 } a_1, a_2, ..., a_T $$

**Reasoning RL**: $T \approx$ 500-30K tokens，单轮生成  
**Agentic RL**: $T \approx$ 100K-1M tokens，多轮交互

---

## 二维分类体系

### 粒度 × 方法论

| | [[PPO\|MC]] | [[GAE\|TD]] | LLM-as-Critic | 博弈论 | 信息论 |
|---|:----------:|:-----------:|:-------------:|:-----:|:------:|
| **Token 级** | VinePPO | RED | T-REG | — | — |
| **片段级** | SPO | — | TEMPO | SCAR | — |
| **步骤级** | Math-Shepherd | PURE | CAPO | HICRA | — |
| **Turn 级** | AgentPRM | SWEET-RL | HCAPO, C3 | CCPO | IGPO |
| **多智能体** | M-GRPO | — | LLM-MCA | SHARP | — |

---

## Reasoning RL → Agentic RL：6 个根本挑战

| # | 挑战 | Reasoning RL | Agentic RL | 影响 |
|:-:|:----|:------------:|:----------:|------|
| 1 | **环境转换** | **确定性**（自回归） | **随机性**（API/网页/代码执行） | MC 方法需重放环境交互，成本极高 |
| 2 | **观测性** | 完全 (MDP) | **部分** (POMDP) | 无法区分"决策错误"和"信息不足" |
| 3 | **轨迹长度** | 0.5K-30K tokens | **100K-1M** | 方差 O(T)，T 增 10 倍方差增 10 倍 |
| 4 | **动作类型** | **同质**（全是 token） | **异质**（工具/计划/通信/恢复） | 错误工具选择可致命，格式问题微不足道 |
| 5 | **中间验证** | **可能**（数学步骤可验） | **几乎不可能** | [[PRM]] 无法直接迁移 |
| 6 | **分叉点** | 中等频率 | **稀有但决定性** | 均匀注意力浪费在无关步骤 |

### 三个不能成立的假设

Reasoning RL 方法能工作的三个假设在 Agentic 中全部失效：

1. **确定性转换**: Agent 调用外部 API → 结果不确 `4`定性
2. **单轮生成**: Agent 需要多轮交互（工具调用→观察→推理→下一步）
3. **可验证结果**: 工具调用成功与否 ≠ 最终任务成功

---

## 代表性方法详细分析

### Reasoning RL 方法

**VinePPO** (Token 级, MC):
- 从每个 token 位置 fork K 条"藤蔓"（继续生成到结束）
- 优势：$A_t = R(\tau) - V(s_t)$
- 需要 $O(K \cdot L)$ 额外前向传播
- **无偏但昂贵**

**SPO** (片段级, MC):
- 在子问题之间的"切割点"划分推理链
- 比较共享前缀的轨迹差异 → 片段级 MC 优势
- **MATH-500 +7.6%**, GSM8K +11.0% over GRPO

**PURE** (步骤级, TD):
- 提出"min-form"信用：$V(s_t) = \mathbb{E}[\min_{t' \geq t} r_{t'}]$
- 防止模型在高分步骤后"隐藏"错误
- **比标准 PRM 更鲁棒**

**HICRA** (步骤级, 博弈论):
- 区分**规划 token** 和**程序性 token**
- 规划 token 获得更高信用（它们决定方向）
- **AIME'24 +4.6%**, AIME'25 +5.1% over GRPO

**SCAR** (片段级, 博弈论):
- 使用 **Shapley 值** 分配片段信用
- 满足效率、对称性、零玩家性质
- 计算指数级复杂度 → 需要采样近似

### Agentic RL 方法

**AgentPRM** (Turn 级, MC+TD):
- 使用 **TD+GAE** 替代 MC 标注（样本效率 8×）
- WebShop **+19%**, TextCraft +13.4% over ORM

**SWEET-RL** (Turn 级, TD):
- **特权非对称 critic**: 训练时 critic 可访问标注答案和完整未来轨迹
- Actor 通过 DPO 使用 critic 提供的 turn 级信号
- ColBench **+6.0%** over MT-DPO

**HCAPO / C3 / CCPO** (Turn 级, 反事实):
- 2026 年 3 月的同一周内**三篇独立论文**！
- **HCAPO**: LLM 在想象中做反事实分析（无需环境重放）
- **C3**: Leave-one-out 框架 $c_t = R(\tau) - R(\tau_{-t})$
- **CCPO**: 将轨迹建模为结构因果模型，do-calculus 算 ATE

**GiGPO** (Turn 级, 免 critic):
- Group-in-Group: 外层组比较轨迹（GRPO 式），内层组通过 anchor state 比较步骤
- **ALFWorld +12.6%**, WebShop +9.1% over GRPO

**CARL** (Turn 级, 信息论):
- 通过 **action 熵** 识别分叉点
- 只在高熵动作上聚焦 RL 更新
- **72% 更少梯度更新**无性能损失，HotpotQA +4.9 F1

---

## 性能汇总

| 方法 | 领域 | 基准 | 提升 | 对比基线 |
|:----|:----:|:----:|:----:|:--------:|
| SPO | Reasoning | MATH-500 | **+7.6%** | GRPO |
| HICRA | Reasoning | AIME'24 | **+4.6%** | GRPO |
| CAPO | Reasoning | AIME'24 | **+6.1%** | GRPO |
| GiGPO | Agentic | ALFWorld | **+12.6%** | GRPO |
| AgentPRM | Agentic | WebShop | **+19.0%** | ORM |
| SWEET-RL | Agentic | ColBench | **+6.0%** | MT-DPO |
| CARL | Agentic | HotpotQA | **+4.9 F1** | GRPO (72% 少更新) |

---

## 三大可复用资源

### 1. 结构化论文清单
机器可读格式（CSV/JSON）的 47 篇论文标签数据：
- 分类标签（粒度 × 方法论）
- 基线家族（G=GRPO, P=PPO, D=DPO, O=ORM, T=TD）
- 证据水平（S=强实证, L=有限, A=纯分析）
- arXiv ID 和主要基准

### 2. 报告检查清单
| 项目 | 已有论文覆盖率 |
|:----|:-------------:|
| CA 粒度说明 | ❌ ~40% 缺失 |
| 对比基线指定 | ✅ ~70% |
| 计算成本报告 | ❌ **~80% 缺失** |
| 消融研究 | ✅ ~60% |
| 分叉点分析 | ❌ **~85% 缺失** |
| 跨设置泛化 | ❌ ~70% 缺失 |

### 3. 基准协议
- Reasoning 任务：已知 ground-truth 步骤信用
- Agentic 任务：受控分叉点的合成环境
- 多智能体任务：设计信用结构
- 元数据格式、JSON schema、方法选择决策树

**方法选择决策树**:

| 场景 | 推荐方法 |
|:----|:--------|
| 数学推理 (GSM8K, MATH) | GRPO, PURE, SPO |
| 竞赛级数学 (AIME) | VinePPO, HICRA, CAPO |
| 工具使用 (WebShop, ALFWorld) | GiGPO, AgentPRM |
| 网页导航 (WebArena) | SWEET-RL, HCAPO, IGPO |
| 软件工程 (SWE-bench) | CARL, HCAPO, C3/CCPO |
| 多智能体系统 | M-GRPO, C3, LLM-MCA |
| 计算受限 | GRPO, CARL, iStar, GiGPO |

---

## 五个核心结论

1. **CA 是 LLM RL 的核心挑战** [强实证]：从 1K token → 1M token，CA 从优化便利变为训练必需
2. **Reasoning RL 正在成熟** [强实证]：Token→片段→步骤各级方法有效，PRM + 免 critic 组比较可扩展
3. **Agentic RL 仍在早期** [有限证据]：反事实分析、特权 critic、turn 级 MDP 是新兴方向
4. **LLM-as-Critic 是新范式** [有限证据]：在经典 RL 中没有直接对应物
5. **领域正在加速** [作者综合]：2026 年 3 月一周内 3 篇独立反事实 CA 论文

---

## 关联笔记

### 涉及的核心方法
- [[PPO]], [[DeepSeekMath|GRPO]], [[DPO]] — 基础算法
- [[GSPO]], [[SAPO]] — GRPO 改进
- [[StabilizingRL]] — 形式化框架

---

## 速查卡片

> [!summary] Credit Assignment Survey
> - **核心**: 47 种 LLM RL 信用分配方法综述
> - **分类**: 粒度（Token→Segment→Step→Turn→Multi-agent）× 方法论
> - **洞察**: Reasoning→Agentic 是 6 个维度的根本转变
> - **资源**: 论文清单 + 检查清单 + 基准协议 + 决策树
> - **推荐**: 按场景选择 CA 方法

---

*笔记创建时间: 2026-07-14 | 深度版*
