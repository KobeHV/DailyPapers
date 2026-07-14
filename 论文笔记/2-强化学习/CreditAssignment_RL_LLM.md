---
title: "From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models"
method_name: "CreditAssignment_RL_LLM"
authors: [Chenchen Zhang]
year: 2026
venue: arXiv
tags: [survey, credit-assignment, reinforcement-learning, reasoning, agentic, llm]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2604.09459v1
created: 2026-07-14
---

# 论文笔记：Credit Assignment in RL for LLMs — 综述

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Independent Researcher |
| 日期 | April 2026 |
| 覆盖范围 | 47 种方法（41 核心 + 6 辅助），2024-2026 |
| 链接 | [arXiv](https://arxiv.org/abs/2604.09459) |

---

## 一句话总结

> 首次系统综述 LLM RL 中的**信用分配问题**，覆盖 47 种方法，提出二维分类体系（粒度 × 方法论），揭示 Reasoning RL → Agentic RL 的根本转变。

---

## 核心贡献

1. **二维分类法**: 按粒度（token/segment/step/turn/multi-agent）× 方法论（MC/TD/Model-based/Game-theoretic/Information-theoretic）
2. **核心洞察**: Reasoning RL 信用分配成熟（PRM + 免 Critic），Agentic RL 催生全新方法
3. **三大可复用资源**: 结构化论文清单 + 报告检查清单 + 基准协议

---

## 方法详解

### 分类体系

| | MC | TD | Model-based | 博弈论 | 信息论 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Token 级** | VinePPO | RED | T-REG | — | — |
| **片段级** | SPO | — | TEMPO | SCAR | — |
| **步骤级** | Math-Shepherd | PURE | CAPO | HICRA | — |
| **Turn 级** | AgentPRM | SWEET-RL | HCAPO | CCPO | IGPO |
| **多智能体** | M-GRPO | — | LLM-MCA | SHARP | — |

### Reasoning → Agentic 的 6 个挑战

| 挑战 | Reasoning RL | Agentic RL |
|------|:------------:|:----------:|
| 环境转换 | 确定性 | **随机性** |
| 观测性 | 完全 (MDP) | **部分** (POMDP) |
| 轨迹长度 | 0.5K-30K tokens | **100K-1M** |
| 动作类型 | 同质化 | **异质化** |
| 中间验证 | 可能 | **几乎不可能** |
| 分叉点 | 中等频率 | **稀有但决定性** |
| 难度 | 2/5 | **5/5** |

### 代表性方法性能

| 方法 | 领域 | 提升 |
|------|:----:|:----:|
| SPO | MATH-500 | **+7.6%** over GRPO |
| HICRA | AIME'24 | **+4.6%** over GRPO |
| GiGPO | ALFWorld | **+12.6%** over GRPO |
| AgentPRM | WebShop | **+19.0%** over ORM |
| SWEET-RL | ColBench | **+6.0%** over MT-DPO |
| CARL | HotpotQA | **+4.9 F1** (72%少梯度) |

---

## 关联笔记

### 涉及的核心方法
- [[PPO]], [[DeepSeekMath|GRPO]], [[DPO]]: 基础算法
- [[GSPO]], [[SAPO]]: GRPO 改进

---

## 速查卡片

> [!summary] Credit Assignment Survey
> - **核心**: LLM RL 信用分配 47 方法综述
> - **方法**: 二维分类（粒度 × 方法论）+ 3 可复用资源
> - **结果**: Reasoning RL 成熟，Agentic RL 刚起步
> - **影响**: 未来 CA 论文的参考起点

---

*笔记创建时间: 2026-07-14*
