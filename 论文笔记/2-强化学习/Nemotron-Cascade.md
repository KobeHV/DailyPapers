---
title: "Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models"
method_name: "Nemotron-Cascade"
authors: [Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, Yang Chen, Yangyi Chen, Zhuolin Yang, Zihan Liu, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping]
year: 2025
venue: arXiv
tags: [reinforcement-learning, cascaded-rl, reasoning, code-rl, math-rl, nvidia]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2512.13607v1
created: 2026-07-14
---

# 论文笔记：Nemotron-Cascade — Cascaded RL

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA |
| 日期 | December 2025 |
| 对比基线 | DeepSeek-R1-0528 (671B), Qwen3-8B/14B |
| 链接 | [arXiv](https://arxiv.org/abs/2512.13607) / [Model](https://huggingface.co/collections/nvidia/nemotron-cascade) |

---

## 一句话总结

> 提出**级联域强化学习**（Cascade RL），按 RLHF → IF → Math → Code → SWE 顺序训练，14B 模型超越 DeepSeek-R1-671B，获 IOI 2025 银牌。

---

## 核心贡献

1. **Cascade RL**: 按领域顺序训练替代混合训练，简化基础设施
2. **RLHF 提升推理**: 仅用对齐数据训练就能大幅提升推理能力
3. **IOI 2025 银牌**: 14B 模型 343 分，超越 OpenAI 金牌模型

---

## 方法详解

### Cascade RL 流程

$$\text{SFT} \rightarrow \text{RLHF} \rightarrow \text{IF-RL} \rightarrow \text{Math RL} \rightarrow \text{Code RL} \rightarrow \text{SWE RL}$$

### 各领域 RL 特性

| 领域 | 验证方式 | 延迟 | 响应长度 |
|------|---------|:---:|:--------:|
| RLHF | 奖励模型打分 | 快 | 12K |
| IF-RL | 规则验证 | 快 | 8-16K |
| Math RL | 符号规则 | **毫秒级** | 16-40K |
| Code RL | 执行验证 | **分钟级** | 44-56K |
| SWE RL | 免执行奖励 | 快 | 24-32K |

### 关键发现：RLHF 本身就提升推理

RLHF **不使用任何数学/编程 prompt**，但大幅提升推理性能：
- ArenaHard: **+18~20**
- LiveCodeBench: **+10~11**
- 原因：RLHF 减少冗长和重复生成

---

## 关键结果

### 14B-Thinking vs DeepSeek-R1-0528 (671B)

| Benchmark | Nemotron-14B | DeepSeek-R1 (671B) |
|-----------|:------------:|:------------------:|
| AIME 2024 | **89.7** | 91.4 |
| AIME 2025 | **83.3** | 87.5 |
| LCB v5 | **77.5** | 74.8 |
| LCB v6 | **74.6** | 73.3 |
| SWE-bench Verified | **43.1** | 57.6 |
| Codeforces Elo | **1932** (97.2%) | - |
| **IOI 2025** | **🥈 343分** | - |

在 LiveCodeBench 上 14B **超越** 671B 模型。

---

## 关联笔记

### 对比
- [[DeepSeekMath|GRPO]]: Nemotron 各阶段使用的 RL 算法
- [[SAO]]: 同步期的工作，专注 agentic RL
- [[RewardAnything]]: 奖励模型方向

---

## 速查卡片

> [!summary] Nemotron-Cascade
> - **核心**: 级联域强化学习（SFT→RLHF→IF→Math→Code→SWE）
> - **方法**: 按领域顺序训练，RLHF 本身就提升推理
> - **结果**: 14B 超越 671B，IOI 2025 银牌
> - **影响**: 证明级联训练 + 小模型可超越大模型

---

*笔记创建时间: 2026-07-14*
