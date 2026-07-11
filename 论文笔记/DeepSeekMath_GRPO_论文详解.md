---
title: DeepSeekMath - Pushing the Limits of Mathematical Reasoning in Open Language Models
authors: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
arxiv_id: "2402.03300"
arxiv_date: 2024-04-27
doi: "10.48550/arXiv.2402.03300"
categories: cs.CL, cs.LG
tags:
  - DeepSeek
  - GRPO
  - 数学推理
  - RL
  - LLM
  - 预训练
  - Common Crawl
link: https://arxiv.org/abs/2402.03300
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 核心贡献

1. **DeepSeekMath 7B** — 在 DeepSeek-Coder-Base-v1.5 7B 基础上继续预训练 **500B tokens**（其中 **120B 数学 token** 来自 Common Crawl），MATH 基准达到 **51.7%**（Top1），逼近 GPT-4 和 Gemini-Ultra
2. **数据筛选管道** — 用 fastText 分类器从 Common Crawl 迭代筛选高质量数学网页（4 轮迭代）
3. **GRPO（Group Relative Policy Optimization）** — PPO 的变体，去掉 critic 模型，用 group scores 估算 baseline，大幅降低训练资源

---

## 1. 数据收集与处理

### 迭代式数据筛选管道

```
Math Seed (OpenWebMath)
        ↓
 ① 训练 fastText 分类器（正例: OpenWebMath 50万条 / 负例: Common Crawl 50万条）
        ↓
 ② 从去重后的 Common Crawl（40B HTML 页面）召回数学网页
        ↓
 ③ 按分数排序，保留 Top-K（第1轮保留 40B tokens 量）
        ↓
 ④ 识别数学相关域名（>10% 页面被召回的域名），人工标注 URL 路径
        ↓
 ⑤ 将标注的 URL 未召回页面加入种子集，迭代训练更强分类器
        ↓
 4 轮迭代后 → 35.5M 数学网页，总计 120B tokens
```

- 第 4 轮迭代发现约 **98%** 的数据已在第 3 轮被收集，停止迭代
- 防污染：10-gram 精确匹配过滤基准测试题

### 数据组成（DeepSeekMath-Base 7B 训练）

| 数据源 | 比例 |
|--------|------|
| DeepSeekMath Corpus（数学网页） | 56% |
| AlgebraicStack（数学代码） | 4% |
| arXiv | 10% |
| GitHub Code | 20% |
| 通用自然语言（中英文 Common Crawl） | 10% |

### 与现有语料对比

| 语料 | 规模 | GSM8K | MATH | CMATH |
|------|------|-------|------|-------|
| 无数学训练 | — | 2.9% | 3.0% | 12.3% |
| MathPile | 8.9B | 2.7% | 3.3% | 11.5% |
| OpenWebMath | 13.6B | 3.7% | 31.3% | 8.9% |
| Proof-Pile-2 | 51.9B | 3.7% | 43.8% | 19.9% |
| **DeepSeekMath Corpus** | **120.2B** | **23.8%** | **13.6%** | **41.5%** |

> 测试条件：DeepSeekLLM 1.3B 基座模型，few-shot CoT 评测

---

## 2. GRPO 算法详解

### PPO 的问题

PPO 框架下需要同时维护：policy model + value model（critic）+ reward model。value model 与 policy model 规模相当，带来巨大显存和计算开销。

### GRPO 的创新

对于每个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{old}}$ 采样一组输出 $\{o_1, o_2, ..., o_G\}$，然后：

1. 用 reward model 对每个输出打分 → 得到 $r = \{r_1, r_2, ..., r_G\}$
2. 对组内 reward 做归一化：$\tilde{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$
3. 所有 token 的 advantage $\hat{A}_{i,t}$ = 归一化后的 reward（outcome supervision）或逐步累积（process supervision）
4. 优化目标最大化 GRPO objective，并直接在 loss 中加入 KL 散度正则化

### GRPO vs PPO 对比

| 方面 | PPO | GRPO |
|------|-----|------|
| 需要 value model | ✅ 是，单独训练 | ❌ 否，组内评分代替 |
| Baseline 来源 | 学出来的 value function | 组内平均 reward |
| KL 惩罚 | 加在 reward 中 | 直接加在 loss 中 |
| 显存占用 | 高 | 低 |
| Reward 适配 | 一般 | 好（reward model 本身就是在比较中训练） |

### 三种监督方式

| 方式 | Advantage 计算 | 效果 |
|------|---------------|------|
| **OS**（Outcome Supervision） | 仅末尾 token 获得归一化 reward，其余相同 | 基线 |
| **PS**（Process Supervision） | 每一步奖励，累加后续步骤归一化 reward | **更优** |
| **Iterative RL** | 迭代更新 reward model + policy model | 显著提升 |

---

## 3. 实验结果

### DeepSeekMath-Base 7B（基座模型）

| 基准 | DeepSeekMath 7B | Mistral 7B | Llemma 34B | Minerva 540B |
|------|:-:|:-:|:-:|:-:|
| GSM8K | **64.2%** | 40.3% | 54.0% | 58.8% |
| MATH | **36.2%** | 14.3% | 25.3% | 33.6% |
| SAT | **84.4%** | 71.9% | 71.9% | — |
| MMLU-STEM | **56.5%** | 51.1% | 52.9% | 63.9% |
| CMATH | **71.7%** | 44.9% | 56.1% | — |

> ✅ 7B 模型在 MATH 上超过 540B Minerva，说明数据质量比模型规模更重要

### DeepSeekMath-Instruct 7B / DeepSeekMath-RL 7B

| 基准 | Instruct 7B | RL 7B | 提升 |
|------|:-:|:-:|:-:|
| GSM8K (CoT) | 82.9% | **88.2%** | +5.3% |
| MATH (CoT) | 46.8% | **51.7%** | +4.9% |
| CMATH | 84.6% | **88.8%** | +4.2% |
| GSM8K+Python | 83.7% | 86.7% | +3.0% |
| MATH+Python | 57.4% | **58.8%** | +1.4% |

> DeepSeekMath-RL **仅用 GSM8K + MATH 的 CoT 格式 SFT 数据** 做 RL，在所有基准上（包括 OOD 如 CMATH）都有提升

---

## 4. 统一范式（Unified Paradigm）

论文提出了统一框架来理解不同训练方法：

$$\nabla J_\mathcal{A}(\theta) = \mathbb{E}\left[\frac{1}{|o|}\sum_{t=1}^{|o|} \mathcal{A}(q, o, r, \theta) \nabla \log \pi_\theta(o_t | q, o_{<t})\right]$$

三个关键组件：

| 组件 | 说明 |
|------|------|
| **Data Source** $\mathcal{D}$ | 训练数据来源（offline/online） |
| **Reward Function** $\mathcal{R}$ | 奖励信号来源（rule/model） |
| **Algorithm** $\mathcal{A}$ | 梯度系数——惩罚或强化的大小 |

### 各方法的差异

| 方法 | Data Source | Reward | 梯度系数 |
|------|-------------|--------|---------|
| SFT | 人工标注 | 人工选择 | 固定 = 1 |
| RFT | SFT 模型采样 (offline) | Rule（答案正确性） | 正确=1，错误=0 |
| DPO | SFT 模型采样 (offline) | Model / Rule | 偏好对比系数 |
| Online RFT | 实时策略采样 (online) | Rule | 同 RFT |
| PPO | 实时策略采样 (online) | Reward Model | GAE advantage |
| **GRPO** | **实时策略采样 (online)** | **Reward Model** | **组内归一化 advantage + KL** |

### 关键发现

- **Online > Offline**：Online RFT 显著优于 RFT（中后期差距拉大）
- **动态梯度系数 > 固定系数**：GRPO > Online RFT（因为 GRPO 对正确/错误响应有不同的强化/惩罚力度）
- **Process > Outcome**：GRPO+PS > GRPO+OS（细粒度逐步骤监督）
- **Iterative RL** 显著提升（第 1 轮迭代效果最明显）

---

## 5. RL 为何有效？

通过分析 Maj@K 和 Pass@K：

- **RL 提升 Maj@K，但不提升 Pass@K**
- 说明 RL 并没有提升模型**能力上限**，而是让输出分布**更稳健**（即把正确答案从 Top-K 中推到 Top-1）
- 这揭示了 SFT 模型中存在 **reasoning misalignment**（能答对但总输出错的）

---

## 6. 重要实验发现

### 代码训练提升数学推理

| 训练方式 | GSM8K | MATH | CMATH |
|----------|------|------|-------|
| 仅数学训练 150B | 19.1% | 14.4% | 37.2% |
| 代码 400B → 数学 150B | **21.9%** | **15.3%** | 39.7% |
| 通用 400B → 数学 150B | 2.9% | 3.2% | 14.8% |

> 代码预训练 → 显著提升数学推理（无论是否使用工具）

### arXiv 论文对数学推理无效

- 在 DeepSeekLLM 1.3B 和 DeepSeek-Coder 7B 上测试
- MathPile（85%+ arXiv）和 ArXiv-RedPajama 均无明显提升，甚至下降
- 局限性：未测试组合效果、大模型规模效果、特定任务效果

---

## 7. 局限性与未来方向

### 局限性
- 几何和定理证明能力弱于闭源模型（可能存在数据选择偏差）
- Few-shot 能力不如 GPT-4（GPT-4 能用 few-shot 提升性能，DeepSeekMath 的 few-shot 与 zero-shot 接近）

### 未来方向

**Data Source：**
- 使用 OOD 问题提示
- 结合树搜索采样策略（Tree-of-Thoughts）
- 改进推理效率技术（推测解码）

**Algorithm：**
- 对抗噪声奖励信号的鲁棒 RL（Weak-to-Strong 对齐）

**Reward Function：**
- 增强 reward model 的泛化能力
- 反映 reward model 的不确定性
- 构建高质量 process reward model

---

## 与相关工作的关系

| 方法 | 关系 |
|------|------|
| RFT | GRPO 的前驱，SFT + 拒绝采样 |
| DPO | 同为离线偏好优化，但使用 pairwise loss |
| PPO | GRPO 的直接前身，去掉 critic 就是 GRPO |
| Online RFT | 与 GRPO 的主要区别在于梯度系数设计 |
| Math-Shepherd | 提供过程监督的思路 |
| WizardMath | PPO 做数学推理的早期工作 |
| InstructGPT | RLHF 框架的源头 |

---

## 总结

DeepSeekMath 的核心价值在于：
1. 证明了 **高质量数据 + 中等规模模型** 可以在数学推理上超越 77× 更大的模型
2. **GRPO** 成为后续 DeepSeek 系列（DeepSeek-R1 等）的重要基础
3. 提供了 RL 方法的统一视角，揭示了 RL 提升 LM 数学推理的本质机制