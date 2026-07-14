---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
method_name: "DeepSeekMath"
authors: [Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo]
year: 2024
venue: arXiv
tags: [mathematical-reasoning, grpo, reinforcement-learning, data-curation, open-model, deepseek]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2402.03300v1
created: 2026-07-14
updated: 2026-07-14
aliases: [DeepSeekMath, GRPO, Group Relative Policy Optimization]
---

# 论文笔记：DeepSeekMath — GRPO 的提出

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | DeepSeek-AI / 清华大学 / 北京大学 |
| **作者** | Zhihong Shao*, Peiyi Wang*, Qihao Zhu*, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo* (\* 在 DeepSeek 实习期间完成) |
| **发表** | arXiv:2402.03300, 2024年4月 |
| **引用量** | > 1,500 |
| **对比基线** | Gemini-Ultra, GPT-4, Minerva 540B, Llemma |
| **链接** | [arXiv](https://arxiv.org/abs/2402.03300) \| [GitHub](https://github.com/deepseek-ai/DeepSeek-Math) |

---

## 一句话总结

> 从 Common Crawl 提取 120B 数学 token + 提出 **GRPO（无价值网络的 PPO 变体）**，7B 模型 MATH 达 51.7% 逼近 GPT-4。

---

## 两大核心贡献

| 贡献 | 解决的问题 | 方法 |
|:----:|-----------|------|
| **1. DeepSeekMath Corpus** | 缺乏高质量数学预训练数据 | 4 轮迭代分类器从 Common Crawl 召回 |
| **2. GRPO** | PPO 需要价值网络导致显存翻倍 | 用组内奖励归一化替代价值网络 |

---

## 贡献 1：DeepSeekMath Corpus — 数学数据筛选

### 数据来源与规模

从 Common Crawl（~40B 去重网页）中提取数学相关内容：

| 语料库 | Token 数 | 相比 Minerva | 相比 OpenWebMath |
|--------|:--------:|:-----------:|:----------------:|
| **DeepSeekMath Corpus** | **120B** | ~7× 更大 | ~9× 更大 |

### 迭代数据筛选流程

```
Common Crawl (40B 网页)
    │
    ▼
Step 1: URL 去重 + 近似去重
    │
    ▼
Step 2: fastText 二元分类器（数学 vs 非数学）
    ├── 正例：OpenWebMath 50 万文档
    └── 负例：随机 Common Crawl 50 万文档
    │
    ▼
Step 3: 按分类器得分排序，保留 top-k
    │
    ▼
Step 4: 人工审核→标记数学域名→添加到种子集→重新训练
    │
    ▼ (迭代 4 次)
最终：3550 万数学网页，120B token
```

**迭代效果**：第 1 轮召回 40B → 第 4 轮召回 120B（覆盖约 98% 可达数学数据）

**分类器配置**:
- 模型：fastText，256 维向量
- 学习率：0.1
- n-gram：最大 3
- epoch：3

### 去污染

过滤包含 GSM8K / MATH / CMATH / AGIEval 等基准测试文本的网页：
- ≥10-gram 精确匹配 → 移除
- 短文本：精确匹配

---

## 贡献 2：GRPO (Group Relative Policy Optimization)

### PPO 的痛点

PPO 同时维护四个模型：
1. 策略模型 $\pi_\theta$
2. 参考模型 $\pi_{\text{ref}}$（用于 KL 计算）
3. 奖励模型 $r_\varphi$
4. **价值模型 $V_\psi$** ← 和策略模型一样大！

价值模型需要：
- 额外的内存和计算（双倍显存）
- 通过 GAE 计算优势函数
- 处理每个 token 的奖励信号

### GRPO 的核心改进：去价值网络

**PPO 的优势计算**（需要价值网络）：

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**GRPO 的优势计算**（不需价值网络）：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, ..., r_G\})}{\text{std}(\{r_1, r_2, ..., r_G\})}$$

其中 $G$ 个输出 $\{o_1, ..., o_G\}$ 从旧策略 $\pi_{\theta_{\text{old}}}$ 采样。

### GRPO 完整目标函数

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)\right]$$

$$\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\left\{
\min\left[\underbrace{\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}}_{\text{token 级重要性比}} \hat{A}_{i,t},\ \text{clip}\left(\cdots, 1-\epsilon, 1+\epsilon\right)\hat{A}_{i,t}\right]
- \beta\underbrace{\mathbb{D}_{KL}[\pi_\theta \parallel \pi_{\text{ref}}]}_{\text{KL 散度正则项}}
\right\}$$

### PPO vs GRPO 详细对比

| 方面 | PPO | GRPO |
|------|:---:|:----:|
| **价值网络** | 需要（增加 1× 内存） | **不需要** |
| **优势计算** | GAE（$V$ 函数 + TD 误差） | **组内归一化**（均值/标准差） |
| **KL 项** | 加入奖励信号 $r_t$ | **直接加入损失** |
| **每个 prompt 采样** | 1 个响应 | **G 个响应**（组） |
| **奖励信号来源** | 奖励模型 + KL 惩罚 | 奖励模型（KL 独立） |
| **训练稳定性** | 对超参数敏感 | 更稳定 |
| **计算复杂度** | 高（4 模型） | **低**（3 模型） |

### 无偏 KL 散度估计

GRPO 使用 Schulman (2020) 的无偏估计器：

$$\mathbb{D}_{KL}[\pi_\theta \parallel \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - \log\frac{\pi_{\text{ref}}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta}(o_{i,t}|q, o_{i,<t})} - 1$$

该估计器保证为正，且不需要期望操作。

### 过程监督 GRPO (Process Supervision)

对每个步骤 $j$ 的奖励进行独立归一化：

$$\widetilde{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$$

每个 token 的优势是其**后续所有步骤**归一化奖励之和：

$$\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \widetilde{r}_i^{\text{index}(j)}$$

### 迭代 GRPO 算法

```
Algorithm: Iterative GRPO

输入: 初始策略 π_init, 奖励模型 r_φ, 训练数据 D, 超参数 ε, β, μ
输出: 优化后的策略 π_θ

1: θ ← θ_init
2: for iteration = 1 to I do:
3:     π_ref ← π_θ                          // 冻结参考模型
4:     for step = 1 to M do:
5:         从 D 采样批次 D_b
6:         π_θ_old ← π_θ                     // 保存旧策略
7:         for each q in D_b:
8:             采样 G 个输出 {o_i} ~ π_θ_old(·|q)
9:             计算奖励 {r_i} (通过 r_φ)
10:            计算优势 A_i (组归一化)
11:        end for
12:        for µ 次内循环:
13:            最大化 GRPO 目标更新 θ
14:        end for
15:     end for
16:     使用回放机制（保留 10% 历史数据）更新 r_φ
17: end for
```

---

## 实验设计

### 预训练 (DeepSeekMath-Base 7B)

| 项目 | 配置 |
|------|------|
| 初始化 | DeepSeek-Coder-Base-v1.5 7B |
| 总训练 token | 500B |
| 数据分布 | 56% Math Corpus + 4% AlgebraicStack + 10% arXiv + 20% Code + 10% NL |
| 优化器 | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| 学习率 | 峰值 4.2e-4，多步退火 |
| 批量大小 | 1000 万 token，上下文 4K |
| 硬件 | HAI-LLM 框架 |

### 监督微调 (DeepSeekMath-Instruct 7B)

| 项目 | 配置 |
|------|------|
| 数据量 | 776K 样本（CoT + PoT + 工具集成） |
| 训练步数 | 500 步 |
| 批量大小 | 256 |
| 学习率 | 5e-5（恒定） |

### 强化学习 (DeepSeekMath-RL 7B)

| 项目 | 配置 |
|------|------|
| 基础模型 | DeepSeekMath-Instruct 7B |
| 训练数据 | 14.4 万 CoT 格式问题（仅 GSM8K + MATH） |
| 奖励模型 | 基于 DeepSeekMath-Base 7B |
| 策略学习率 | 1e-6 |
| KL 系数 | 0.04 |
| 组大小 G | 64 |
| 最大长度 | 1024 |
| 批量大小 | 1024 |

---

## 实验结果

### 核心数学推理

| 模型 | GSM8K | MATH |
|------|:-----:|:----:|
| DeepSeekMath-Base 7B | 64.2% | 36.2% |
| DeepSeekMath-Instruct 7B | 82.9% | 46.8% |
| **DeepSeekMath-RL 7B** | **88.2%** | **51.7%** |
| + Self-Consistency (64) | — | **60.9%** |
| Minerva 540B (对比) | 58.8% | 33.6% |

> DeepSeekMath-Base 7B（36.2% MATH）**在 1/77 参数量下超越 Minerva 540B（33.6%）**。

### 预训练数据消融 (1.3B 模型, 150B token)

| 语料 | Token 数 | GSM8K | MATH | SAT | MMLU-STEM |
|------|:--------:|:-----:|:----:|:---:|:---------:|
| 无数学数据 | 0 | 2.9% | 3.0% | 15.6% | 19.5% |
| MathPile | 8.9B | 2.7% | 3.3% | 12.5% | 15.7% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 31.3% | 29.6% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 43.8% | 29.2% |
| **DeepSeekMath Corpus** | **120.2B** | **23.8%** | **13.6%** | **56.3%** | **33.1%** |

> 数据质量和规模都很重要。DeepSeekMath Corpus 在同等条件下显著超越所有公开数学语料。

### GRPO 与各方法对比 (1.3B 消融)

**不同 RL 方法效果**:

| 方法 | 需要价值网络 | 需要 on-policy 采样 | MATH 性能 |
|------|:-----------:|:------------------:|:---------:|
| SFT | ❌ | ❌ | 基线 |
| RFT (Rejection Finetuning) | ❌ | ❌ | + |
| Online RFT | ❌ | ✅ | ++ |
| **GRPO** | **❌** | ✅ | **+++** |
| PPO | ✅ | ✅ | +++ (更高资源) |

**关键发现**:
- 在线 RFT 显著优于离线 RFT（差距随训练增大）
- GRPO 优于 Online RFT（因为 GRPO 差异化地增强/抑制不同响应）
- 过程监督 GRPO+PS 优于结果监督 GRPO+OS

### RL 为何有效？

**重要分析**: RL 提升了 **Maj@K**（多数投票）但**不提升 Pass@K**：

这意味着 RL **不**增强模型的"基础能力"（生成正确答案的概率），而是**使输出分布更稳健**（将正确响应推到 Top-k 范围内，错误响应被抑制）。这与 SFT 模型的"不对齐问题"一致—SFT 模型知道正确答案但不能稳定生成。

---

## 批判性分析

### 优点
1. **GRPO 是 PPO 的重要简化**: 去掉价值网络使内存降低 ~50%，使更大规模 RL 成为可能
2. **数学数据筛选流程可复现**: 4 轮迭代 + fastText 的高效方法
3. **开源贡献**: 模型、代码、数据均开源

### 局限性
1. **Token 级 IS ratio 有理论缺陷**: GRPO 的 token 级重要性采样在统计上不合法（[[GSPO]] 指出此问题）
2. **对组大小敏感**: 组大小 G 显著影响性能，需要调参
3. **只用于数学**: 有效性主要在数学推理上验证，未推广到更通用的 RLHF
4. **GRPO 的训练不稳定**: 在长序列中 GRPO 可能崩溃（[[GSPO]] 和 [[StabilizingRL]] 都讨论此问题）

### GRPO 的改进演化

```
GRPO (DeepSeekMath, 2024)
  ├── 问题: Token 级 IS 理论缺陷
  │   └── GSPO (阿里, 2025): 序列级 IS ratio
  ├── 问题: 硬裁剪浪费信号
  │   └── SAPO (阿里, 2025): 软门控
  ├── 问题: 异步场景不兼容
  │   └── SAO (清华, 2026): 单 rollout 异步 RL
  └── 问题: 训练不稳定根源
      └── StabilizingRL (阿里, 2025): 形式化框架 + R2/R3
```

---

## 关联笔记

### 前置基础
- [[PPO]] — GRPO 的基础算法
- [[RLHF]] — 对齐框架

### 后续改进
- [[GSPO]] — 序列级 importance ratio 改进 GRPO
- [[SAPO]] — 软门控替代硬裁剪
- [[SAO]] — 异步单 rollout 优化
- [[StabilizingRL]] — 形式化框架 + MoE 路由重放

---

## 速查卡片

> [!summary] DeepSeekMath / GRPO
> - **核心**: 无价值网络的组相对策略优化
> - **方法**: 组内奖励归一化 ($\hat{A}_i = (r_i-\mu)/\sigma$) + KL 直接加入损失
> - **数据**: 从 Common Crawl 经 4 轮迭代精选 120B 数学 token
> - **结果**: 7B MATH 51.7% → 60.9% (SC@64)，超越 Minerva 540B
> - **影响**: GRPO 成为 LLM RL 后训练的主流方法

---

*笔记创建时间: 2026-07-14 | 深度版*
