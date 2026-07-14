---
title: "Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models"
method_name: "Nemotron-Cascade"
authors: [Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, Yang Chen, Yangyi Chen, Zhuolin Yang, Zihan Liu, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping]
year: 2025
venue: arXiv
tags: [reinforcement-learning, cascaded-rl, reasoning, code-rl, math-rl, nvidia, ioimedical]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2512.13607v1
created: 2026-07-14
updated: 2026-07-14
aliases: [Nemotron-Cascade, Cascade RL, Nemotron]
---

# 论文笔记：Nemotron-Cascade — Cascaded RL

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | NVIDIA |
| **作者** | Boxin Wang, Chankyu Lee, Nayeon Lee, Sheng-Chieh Lin, Wenliang Dai, Yang Chen, Yangyi Chen, Zhuolin Yang, Zihan Liu, Mohammad Shoeybi, Bryan Catanzaro, **Wei Ping** (lead) |
| **发表** | arXiv:2512.13607, 2025年12月 |
| **对比基线** | DeepSeek-R1-0528 (671B), Qwen3-8B/14B |
| **模型发布** | [HuggingFace Collection](https://huggingface.co/collections/nvidia/nemotron-cascade) |
| **链接** | [arXiv](https://arxiv.org/abs/2512.13607) |

---

## 一句话总结

> **级联域强化学习**（Cascade RL）按领域顺序训练：RLHF→IF→Math→Code→SWE，14B 模型超越 DeepSeek-R1-671B 并获得 IOI 2025 银牌 🥈

---

## 核心洞察

### 不同领域的 RL 特性差异

| 领域 | 验证方法 | 延迟 | 响应长度 | 适合级联的理由 |
|:----|---------|:----:|:--------:|:-------------|
| **RLHF** | 奖励模型打分 | 快 | 12K | 广泛提升基础能力 |
| **IF-RL** | 规则验证 | 快 | 8-16K | 增强指令遵循 |
| **Math RL** | 符号规则验证 | **毫秒级** | 16K→40K | 快速迭代 |
| **Code RL** | 执行验证 | **分钟级** | 44-56K | 慢速但精准 |
| **SWE RL** | 免执行奖励 | 快 | 24-32K | 需要代码基础 |

> 混合所有领域训练的问题：Math 验证毫秒级，Code 验证分钟级 → 
整个 batch 等最慢 → GPU 空转。级联训练完美解决此问题。

### Cascade RL 的核心优势

1. **简化基础设施**: 各领域独立训练，无需复杂调度
2. **天然抗遗忘**: RL 的抗遗忘能力比 SFT 强（策略依赖的数据分布 + 连续奖励信号）
3. **领域特定优化**: 每个阶段可用不同的超参数、奖励函数、长度限制

---

## 方法详解

### 完整的级联流程

$$\boxed{\text{SFT}} \rightarrow \boxed{\text{RLHF}} \rightarrow \boxed{\text{IF-RL}} \rightarrow \boxed{\text{Math RL}} \rightarrow \boxed{\text{Code RL}} \rightarrow \boxed{\text{SWE RL}}$$

### Stage 0: SFT（监督微调）

**两阶段 SFT**（逐步扩展上下文）：

| 阶段 | 上下文长度 | 数据内容 |
|:---:|:---------:|---------|
| Stage 1 | 16K | 通用数据 + 数学/科学/代码推理（thinking 模式） |
| Stage 2 | 32K | Stage 1 数据 + 长链推理 + 工具使用 + SWE |

**Chat 模板创新**: 使用 `/think` 和 `/no_think` 标签，支持全局和逐轮控制思考模式。

**SFT 数据规模**: 

| 数据类型 | 样本数 | Token 数 |
|---------|:------:|:--------:|
| 通用领域 | 2.8M | 3.2B |
| 数学 (多版采样) | 353K prompts | 2.77M samples |
| 编程 | 172K | 1.42M |
| 科学 | 226K | — |
| 工具调用 | 310K | — |
| SWE 修复 | 127K | — |

**教师模型**: DeepSeek-R1-0528 (thinking) + DeepSeek-V3-0324 (non-thinking)

### Stage 1: RLHF

| 项目 | 配置 |
|:----|------|
| 算法 | GRPO（严格 on-policy） |
| 最大长度 | 12K，不过滤超长 |
| 训练步骤 | 800-900 |
| 模式 | **一半 thinking + 一半 non-thinking** |
| 惩罚 | 代码混用惩罚（language mixing） |

**关键发现**: RLHF **大幅提升推理能力**，尽管完全没用数学/编程数据！

| 基准 | SFT | +RLHF | 提升 |
|:----|:---:|:-----:|:----:|
| ArenaHard | — | **+18~20** | 显著 |
| LiveCodeBench | — | **+10~11** | 显著 |
| AIME | — | + | 有提升 |

**原因**: RLHF 减少冗长和重复 → 模型更简洁有效的推理。

### Stage 2: IF-RL（指令遵循）

- 两阶段递进（IFEval → IF-Bench-Train）
- Unified 模型只在 non-thinking 模式下应用（防 reward hacking）
- 使用动态过滤（移除 0% 和 100% 正确率的问题）

### Stage 3: Math RL

| 项目 | 配置 |
|:----|------|
| 数据 | 18K 过滤后的数学问题 |
| 验证器 | 规则验证器（检查 boxed answer） |
| 长度递进 | **24K → 32K → 40K**（压缩→扩展→长链） |
| 温度 | 1.0 |
| Batch Size | 128 |
| 总步数 | ~500 |
| 动态过滤 | 每轮移除 0%/100% 正确率问题 |

**长度递进策略**：压缩阶段（24K）将不同初始化模型统一到 ~16K 推理长度，使后续阶段可跨模型工作。

### Stage 4: Code RL

| 项目 | 配置 |
|:----|------|
| 数据 | 9.8K 过滤后的编程问题 |
| 验证器 | 单元测试（1=全部通过, 0=否则） |
| 异步奖励计算 | **1172.4s → 416.2s**（-65%） |
| 最大长度 | 44K-56K |
| 温度 | 1.0 |
| 学习率 | 4e-6 |
| 代码混用惩罚 | 0（数学时的 -1 会伤害代码性能） |

### Stage 5: SWE RL（软件工程）

**免执行奖励**: 基于补丁相似度（词汇 + 语义）
- 词汇相似度：BLEU / 编辑距离
- 语义相似度：**Kimi-Dev-72B** 的 YES token 概率

| 项目 | 配置 |
|:----|------|
| 上下文 | 16K → 24K (8B), 32K (14B) |
| 每 prompt rollout | 16 |
| 数据 | ground-truth 定位 OR 检索文件 |

---

## 关键结果

### 主实验结果

| 基准 | Qwen3-8B | **Nemotron-8B** | Qwen3-14B | **Nemotron-14B-T** | DeepSeek-R1 (671B) |
|:----|:--------:|:--------------:|:--------:|:------------------:|:------------------:|
| **MMLU** | 83.0 | **83.7** | 84.9 | **85.1** | 89.9 |
| **MMLU-Pro** | 75.1 | **75.7** | 77.6 | **77.0** | 85.0 |
| **GPQA-Diamond** | 62.0 | **66.5** | 64.0 | **69.6** | 81.0 |
| **ArenaHard** | 85.8 | **87.9** | 91.7 | **89.5** | 95.1 |
| **IFEval (strict)** | 85.0 | **90.2** | 85.4 | **81.9** | 84.1 |
| **AIME 2024** | 76.0 | **89.5** | 79.3 | **89.7** | 91.4 |
| **AIME 2025** | 67.3 | **80.1** | 70.4 | **83.3** | 87.5 |
| **LCB v5** | 61.2 | **74.3** | 65.2 | **77.5** | 74.8 |
| **LCB v6** | 58.3 | **71.1** | 63.5 | **74.6** | 73.3 |
| **LCB Pro Easy** | 46.1 | **65.7** | 53.6 | **68.9** | 63.9 |
| **LCB Pro Med** | 2.2 | **6.4** | 2.6 | **10.5** | 7.0 |
| **SWE-bench Verified** | 20.5 | **37.2** | 27.4 | **43.1** | 57.6 |

> **14B 在 LiveCodeBench 所有子集上超越 671B DeepSeek-R1** 🏆

### IOI 2025 成绩

| 指标 | Nemotron-14B-T |
|:----|:--------------:|
| **总分** | **🥈 343.37 分（银牌）** |
| **Problem 2 (Triples)** | **90.37** 分 |
| OpenAI IOI-gold | 75.29 分（Problem 2） |
| DeepSeek-V3.2-Speciale | 82 分（Problem 2） |

### Codeforces Elo

| 模型 | Elo | 百分位 |
|:----|:---:|:------:|
| Nemotron-8B | 1789 | 95.7% |
| **Nemotron-14B-T** | **1932** | **97.2%** |

### 级联各阶段性能变化

RLHF → IF-RL → Math RL → Code RL → SWE RL 过程中，每个阶段：

- **主要目标提升显著**
- **其他基准极少下降**
- **Code RL 后的 SWE RL 可额外提升 SWE-bench +5.6%**

---

## 训练超参数汇总

| 阶段 | 算法 | 学习率 | Batch | 长度 | 步数 | 温度 |
|:----|:----:|:-----:|:-----:|:----:|:---:|:----:|
| RLHF | GRPO | 1e-6 | — | 12K | 800-900 | — |
| IF-RL | GRPO | 1e-6 | — | 8-16K | — | — |
| Math RL | GRPO | 1e-6 | 128 | 24→40K | ~500 | 1.0 |
| Code RL | GRPO | 4e-6 | — | 44-56K | — | 1.0 |
| SWE RL | GRPO | — | — | 16-32K | — | — |

---

## 批判性分析

### 优点
1. **工程创新**: Cascade RL 的理念简洁实用，解决了多领域训练的核心矛盾
2. **14B 超越 671B**: 证明**算法/数据 > 模型规模**
3. **RLHF 提升推理的发现**: 对 RL 社区有重要启示
4. **完整的开源**: 模型和数据全部开源

### 局限性
1. **高度工程化**: Cascade RL 更多是工程方案而非算法创新
2. **顺序固定**: 级联顺序经过精心设计，新领域加入需要重新实验
3. **计算量**: 5 阶段级联总计算量巨大
4. **SWE RL 依赖外部模型**: 使用 Kimi-Dev-72B 作为奖励信号

---

## 关联笔记

### 对比方法
- [[DeepSeekMath|GRPO]]: Nemotron 各阶段使用的 RL 算法
- [[SAO]]: Agentic RL 方案，不同路径
- [[RewardAnything]]: 奖励模型泛化

---

## 速查卡片

> [!summary] Nemotron-Cascade
> - **核心**: 级联域强化学习
> - **顺序**: SFT → RLHF → IF-RL → Math RL → Code RL → SWE RL
> - **结果**: 14B 超越 671B，IOI 2025 🥈，Codeforces 97.2%
> - **关键发现**: RLHF 使用非推理数据提升推理能力
> - **影响**: 证明级联训练 + 小模型可超越大模型

---

*笔记创建时间: 2026-07-14 | 深度版*
