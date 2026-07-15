# RL 强化学习提升数学/教育解题 —— 论文深度阅读笔记

> 调研时间范围：2025 年 1 月 – 2026 年 7 月 | 整理日期：2026-07-15
> 深度阅读约 20 篇核心论文，每篇含方法论、实验设计、关键结果和个人评注

---

# 第一部分：奠基性工作

---

## 1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **作者**: DeepSeek-AI（通讯：梁文锋）
- **发表**: arXiv 2025.01 → **Nature 封面** (Vol. 645, 2025.09)
- **论文链接**: arxiv.org/abs/2501.12948

### 1.1 核心问题

能否**仅通过强化学习（无人类推理示范 SFT）**让 LLM 自发产生推理能力？

### 1.2 方法论

#### GRPO (Group Relative Policy Optimization) —— 替代 PPO

GRPO 是整篇论文的技术基石。与标准 PPO 的关键区别在于**不需要 Value Network**，极大降低显存和计算开销：

1. 对每个 prompt 采样 G 个输出（一组）
2. 用**规则验证器**对每个输出打分（数学题 = 答案匹配，编程题 = 单元测试通过）
3. **组内归一化**：用组内均值和标准差归一化得到 advantage → $A_i = \frac{r_i - \mu_{group}}{\sigma_{group}}$
4. 带 clipping 的 Policy Gradient 更新 + KL 惩罚到参考模型

**奖励设计**:
- 准确率奖励：规则匹配（数学答案对/错）
- 格式奖励：确保输出保持 `<think>...</think>` 和 `<answer>...</answer>` 格式

#### DeepSeek-R1-Zero：纯 RL 训练

- 基座：DeepSeek-V3-Base
- 仅用准确率 + 格式奖励，**无人类 CoT 冷启动数据**
- AIME 2024 pass@1：15.6% → **77.9%**（超过普通人类选手）
- majority vote @16：**86.7%**

#### "Aha Moment" 现象

训练约 8200 步时，模型自发出现反思行为：
- "wait, let me check..."
- "Alternatively, we could..."
- 模型学会了**暂停、自我验证、回溯、探索替代方案**

#### DeepSeek-R1：四阶段流水线

R1-Zero 的问题：可读性差、中英混杂、未对齐。R1 的解决：

1. **冷启动 SFT**：数千条人类风格的 CoT 示例微调
2. **推理导向 RL**：大规模 RL（同 R1-Zero）+ 语言一致性奖励
3. **拒绝采样 + SFT**：用 RL 收敛的模型生成 ~600K 推理样本 + ~200K 非推理样本，SFT 基座模型
4. **全场景 RL**：第二轮 RL——推理题用规则奖励，通用题用模型奖励

#### 蒸馏

将 R1 的知识蒸馏到 Qwen/Llama (1.5B–70B)：
- ~800K R1 生成的样本（仅 SFT，不再 RL）
- 蒸馏后的 Qwen-32B 和 Llama-70B**显著超越** OpenAI o1-mini

### 1.3 关键结果

| 基准 | R1 得分 |
|------|---------|
| AIME 2024 pass@1 | **79.8%** |
| MATH-500 | **97.3%** |
| Codeforces rating | **~2029** |
| AIME 2024 pass@64（投票扩展） | **90.0%** |

### 1.4 个人评注

> DeepSeek-R1 是 2025 年影响力最大的 AI 论文之一。它的核心洞见——"Hard problems + Reliable Verifiers + Sufficient RL compute"——定义了整个 RL-for-reasoning 方向。但后续 Spurious Rewards 论文揭示了一个重要问题：GRPO 的 clipping 可能在**放大预训练先验**而非真正"教会"推理。R1 中观察到的"aha moment"到底是从零学会的还是被 RL 从预训练中"挤出来"的，至今仍有争议。

---

## 2. rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking

- **作者**: Xinyu Guan, Li Lyna Zhang 等 (Microsoft Research Asia)
- **发表**: ICML 2025 **Oral**
- **论文链接**: arxiv.org/abs/2501.04519
- **代码**: github.com/microsoft/rStar

### 2.1 核心问题

**小模型（7B）能否不依赖大模型蒸馏，达到 o1 级别的数学推理？**

### 2.2 方法论

#### 三大创新

**① 代码增强 CoT 数据合成**
- MCTS rollout 过程中，每个推理步骤同时生成自然语言 + Python 代码
- 仅保留代码**验证通过**的步骤
- 大规模 MCTS rollout 产逐步验证的推理轨迹

**② Process Preference Model (PPM)**
- 不直接标注步骤分数（噪声太大），改用**成对偏好排名**
- 用 MCTS 的 Q 值区分好步骤 vs 坏步骤
- 训练 PPM 学会"偏好"好的推理步骤

**③ 四轮自进化**
- 策略 SLM 和 PPM 从零开始，**迭代共进化**
- 每轮用当前模型生成更好的训练数据 → 训练更强的下一轮模型
- 数据规模扩展到 **747K 数学题** + 数百万解答

### 2.3 关键结果（Qwen2.5-Math-7B）

| 基准 | 提升 |
|------|------|
| MATH | 58.8% → **90.0%** (+31.2%，超越 o1-preview +4.5%) |
| Phi3-mini (3.8B) | 41.4% → 86.4% |
| AIME 2024 | 解出 **53.3% (8/15)**，位列美国高中数学竞赛者前 20% |

### 2.4 个人评注

> rStar-Math 是**小模型 + 搜索**路线的扛鼎之作。核心优雅之处在于：MCTS 既是推理引擎也是数据工厂——测试时用来搜索，生成的轨迹用来训练。PPM 的成对偏好设计避开了步骤级标注的瓶颈。与 DeepSeek-R1 的"RL 出推理"不同，rStar-Math 用的是"搜索出推理 + 搜索做数据"。两者思想互补。

---

## 3. Spurious Rewards: Rethinking Training Signals in RLVR ⚠️

- **作者**: Rulin Shao, Stella Li 等 (UW / Allen AI / UC Berkeley)
- **发表**: ICML 2026, arXiv: 2506.10947 (2025.06)
- **代码**: github.com/ruixin31/Rethink_RLVR

### 3.1 核心发现（震撼性的）

**用完全虚假的奖励信号训练 Qwen2.5-Math-7B，在 MATH-500 上仍然大幅提升：**

| 奖励类型 | MATH-500 提升 |
|----------|:------------:|
| 真实答案奖励 | **+29.1%** |
| **随机抛硬币奖励** | **+21.4%** |
| **故意错误标签奖励** | **+24.1%** |
| 仅格式奖励（检查 \boxed{}） | +13.8% |
| 1-shot RL | +26.0% |
| 多数投票 | +27.1% |

**随机抛硬币几乎追平真实奖励！**

### 3.2 原因分析

#### 机制一：GRPO Clipping Bias

GRPO 的 clipping 项（ε=0.2）不仅稳定训练，还创造了**隐式熵最小化**效应：
- 即使奖励是随机的，clipping 也持续将策略**集中**到预训练中已有的高概率行为上
- **消融实验**：移除 clipping（三种方法），随机奖励的增益**完全消失** → 增益 100% 来自 clipping

#### 机制二：Code Reasoning 能力

Qwen2.5-Math 独有的"代码推理"行为：
- **RLVR 前**：65% 的回答使用 Python 代码推理（但不执行代码）
- **RLVR 后（即使随机奖励）**：90%+ 使用代码推理
- 模型在自回归过程中精确预测代码输出（如心算 π 的 15 位小数）
- 这个能力在 Llama3、OLMo2 中**基本不存在**

### 3.3 跨模型验证——这是 Qwen 专属现象！

| 模型家族 | 随机奖励效果 |
|----------|:----------:|
| **Qwen2.5-Math** | +21.4% ✅ |
| **Llama3** | 无增益 |
| **OLMo2** | **-5.3% 退化** |
| Llama3 + 格式奖励 | **-7.3% 退化** |

### 3.4 对领域的冲击

- 许多 RLVR 论文（TTRL、1-shot RL 等）**仅基于 Qwen 验证**，结论可能不具普适性
- 后续工作 (Yan et al., "Spurious Rewards Paradox") 认为 RLVR 激活的是**预训练数据污染的"记忆捷径"**
- 未来 RLVR 研究必须**跨多种模型家族验证**

### 3.5 个人评注

> 这是 2025 年最让我重新思考 RLVR 的一篇论文。它不仅揭露了 Qwen 模型家族的特殊性（code reasoning 先验），也揭示了 GRPO clipping 的"隐式熵最小化"机制——你给它随机信号，它也会帮你缩小搜索空间、聚焦已有能力。这带来的核心问题是：**RLVR 的性能提升到底是"学会推理"还是"想起已知"？** 对于教育解题方向，这意味着如果我们用 RLVR 训练解题模型，模型可能只是在"唤醒"预训练语料中的解题模板，而非真正掌握可迁移的推理能力。

---

# 第二部分：RL 算法与训练方法

---

## 4. Open-Reasoner-Zero (ORZ)

- **作者**: Jingcheng Hu, Yinmin Zhang 等
- **发表**: NeurIPS 2025, arXiv: 2503.24290
- **代码**: github.com/openreasoner/openr

### 4.1 核心贡献

**首个完全开源的大规模 RL-from-base-model 推理训练方案**，且方法论极简。

### 4.2 极简方法

- **Vanilla PPO + GAE** (λ=1, γ=1)
- **无 KL 正则**、**无 SFT 冷启动**、**无格式奖励**
- 仅用二元奖励（答案对=1，错=0）
- **不做任何奖励工程**

### 4.3 关键结果（32B）

| 基准 | ORZ | DeepSeek-R1-Zero-32B |
|------|:---:|:---:|
| AIME 2024 | **48.1** | 47.0 |
| MATH500 | **92.2** | 91.6 |
| GPQA Diamond | **55.5** | 55.0 |

- 训练效率约 **1/10** 的 DeepSeek-R1-Zero
- 0.5B 到 32B 全系列有效，跨 Qwen/Llama/Mistral/DeepSeek-Math 家族
- 同样观察到自验证、回溯、"aha moment"

### 4.4 个人评注

> ORZ 的极简主义本身就是一种声明：**你不需要复杂的奖励工程来激发推理**。纯 PPO + 二元奖励就够了。但如果结合 Spurious Rewards 的发现，ORZ 的成功可能也高度依赖基座模型的预训练先验（它的实验同样大量使用 Qwen 系列）。

---

## 5. DAPO: An Open-Source LLM RL System at Scale

- **作者**: Qiying Yu 等 35 位作者（字节跳动/清华/港大）
- **发表**: NeurIPS 2025, arXiv: 2503.14476
- **代码**: github.com/volcengine/verl (verl 框架)

### 5.1 四项关键技术

DAPO 解决的是 naive GRPO 在大规模训练中遇到的熵坍缩、奖励噪声、训练不稳定问题。

| 技术 | 问题 | 解决方案 |
|------|------|---------|
| **Clip-Higher** | 对称 clipping 对低概率 token 不公平（0.01→0.012 vs 0.9→1.08） | 解耦上下界：ε_low=0.2, ε_high=0.28，给探索 token 更多增长空间 |
| **Dynamic Sampling** | 全对/全错 prompt 的 advantage 为 0 → 梯度消失 | 过采样 + 过滤掉 acc=0 或 acc=1 的 prompt，保持有效 batch |
| **Token-Level Loss** | 样本级 loss 对长短序列等权重，惩罚长序列中的重复/乱码不够 | 所有 token 加总再归一化，长序列获得比例更大的影响 |
| **Overlong Reward Shaping** | 截断超长输出造成奖励噪声 | 软惩罚：在 buffer 区内线性从 0 到 -1，而非一刀切 |

### 5.2 AIME 2024 结果

| 配置 | AIME 2024 |
|------|:---------:|
| Naive GRPO | ~30% |
| + Clip-Higher | 38% |
| + Soft Overlong Punishment | 41% |
| + Token-Level Loss | 42% |
| **+ Dynamic Sampling (Full DAPO)** | **50%** |
| 最新复现 | **52%** |

- 训练步数比 DeepSeek-R1-Zero-32B 少 **50%**
- 基座：Qwen2.5-32B
- 完全开源（代码 + 数据集 DAPO-Math-17K + 模型权重）

### 5.3 争议：Dynamic Sampling 真的有用吗？

后续 Comparative Analysis 论文 (arXiv 2512.07611) 系统对比 PPO/GRPO/DAPO 发现：
- DAPO 的 Dynamic Sampling **并未提升性能**，禁用后反而一样/更好
- Clip-Higher 和 Token-Level Loss 可能是 DAPO 增益的主要来源

### 5.4 个人评注

> DAPO 的实际贡献不在于 Dynamic Sampling（可能被证伪），而在于**把 GRPO 的工程细节系统化了**。Clip-Higher 和 Token-Level Loss 确实是 GRPO 的重要改进。DAPO 也是 verl 框架的标志性工作，开源生态价值巨大。

---

## 6. rStar2-Agent: Agentic Reasoning (14B 击败 DeepSeek-R1 671B)

- **作者**: Microsoft Research
- **发表**: arXiv: 2508.20722 (2025.08)
- **代码**: github.com/microsoft/rStar

### 6.1 核心创新：GRPO-RoC (Resample-on-Correct)

标准 RL 在 agentic 环境（模型写代码 → 执行 → 看结果）中的问题：模型写出有 bug 的代码，但偶然得到正确答案 → RL 给满分奖励 → 强化了低效习惯。

GRPO-RoC 的解决方案——**非对称过滤策略**：

1. **过采样**：每题生成 32 条轨迹（2G）
2. **正样本严格筛选**：仅保留"工具调用错误最少 + 格式规范 + 推理清晰"的高质量正确解
3. **负样本均匀保留**：保留错误答案的多样性
4. **零 reward hacking**：不修改奖励函数

### 6.2 三段式训练配方

| 阶段 | 长度限制 | 数据 | 设计意图 |
|------|:------:|------|---------|
| SFT | 2K | 165K 函数调用 + 57K 通用指令 | **刻意不灌长推理样本**，只学指令遵循+JSON 格式+工具使用 |
| RL-1 | 8K | 42K 整数答案题 | "先学简洁"，避免过早固化冗长推理 |
| RL-2 | 12K | 同上 | 长度放开，+5% |
| RL-3 | 12K | 17K 难题 | 专攻硬骨头，+3.6% |

### 6.3 关键结果

| 模型 | AIME24 | AIME25 | HMMT25 |
|------|:------:|:------:|:------:|
| **rStar2-Agent-14B** | **80.6%** | **69.8%** | **52.7%** |
| DeepSeek-R1 (671B) | 79.8% | 70.0% | 44.4% |
| o3-mini (medium) | 79.6% | 77.0% | 53.0% |

- **仅 64 张 MI300X GPU，一周完成 510 步 RL**
- 推理长度约 DeepSeek-R1 的一半
- **泛化到非数学领域**：GPQA-Diamond 60.9%（科学推理）

### 6.4 个人评注

> rStar2-Agent 最让我印象深刻的是它的**哲学转变**——从"think longer"到"think smarter"。模型的行为链是"写代码 → 跑结果 → 根据反馈反思 → 调整思路"，这比纯 CoT 自我对话高效得多。14B 超越 671B，对教育和低资源场景的启示巨大。

---

## 7. The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

- **作者**: Xinyu Zhu, Mengzhou Xia, Danqi Chen 等 (Princeton / UVA)
- **发表**: NeurIPS 2025, arXiv: 2506.01347
- **代码**: github.com/TianHongZXY/RLVR-Decomposed

### 7.1 核心发现

将 RLVR 目标分解为两个独立的信号：

$$\mathcal{L}_{RLVR} = \mathcal{L}_{PSR}(正面强化) + \mathcal{L}_{NSR}(负向惩罚)$$

**仅用负向惩罚（NSR-only）**——只压制错误答案，从不奖励正确答案——**可以匹配甚至超越 PPO/GRPO**。

### 7.2 三项关键实验

**实验一：NSR vs PSR vs Full RL（MATH, Qwen2.5-Math-7B）**

| 方法 | Pass@1 | Pass@256 | 行为 |
|------|:------:|:--------:|------|
| PSR-only | ↑ 提升 | **↓ 退化**（低于基座） | 输出分布坍缩 |
| **NSR-only** | ↑ 提升 | **↑ 保持/提升** | 保留多样性 |
| Full RL (PPO/GRPO) | ↑ 提升 | ↓ 退化 | PSR 效应占主导 |

**实验二：Qwen3-4B 的 latent reasoning 激活**
- Qwen3-4B think mode：MATH Pass@1 = 94.5%
- Qwen3-4B non-think mode：远低于此
- PSR **无法激活** think mode 的潜在能力，甚至退化
- NSR 和 GRPO 成功解锁 think mode 能力（NSR 达到 94.0%，接近 think mode 的 94.5%）

**实验三：梯度分析**
- NSR 通过**压制错误的推理步骤**来工作
- 被压制的概率质量重新分配到模型先验中已有的合理替代路径
- 是"精炼已有知识"而非"引入全新行为"

### 7.3 提出的方法：Weighted-REINFORCE

基于 NSR 被低估的洞察，对 REINFORCE 目标做简单修改——**上调 NSR 贡献权重**：
- 在 MATH、AIME 2025、AMC23 上一致超越 PPO 和 GRPO

### 7.4 个人评注

> 这篇论文的观念冲击不亚于 Spurious Rewards。我们一直认为 RL 让模型变聪明靠的是"奖励正确路径"，但 NSR 的实验表明**压制错误路径才是更重要的机制**，尤其在保持输出多样性方面。这对于教育场景特别有意义：一个好的数学老师，很多时候做的正是帮学生"避免错误的思考方式"，而非直接告诉正确答案。

---

## 8. 1-Shot RLVR: Reinforcement Learning for Reasoning with One Training Example

- **作者**: Yiping Wang 等
- **发表**: NeurIPS 2025 Most Influential Paper
- **代码**: github.com/ypwang61/One-Shot-RLVR

### 8.1 核心发现

**仅用 1 个训练样本 + RLVR** → Qwen2.5-Math-1.5B 在 MATH500 上从 36.0% → **73.6%**

2 个样本 → 74.8%，匹配使用 1200 样本训练的完整效果。

### 8.2 方法细节

#### 数据选择：Historical Variance Score
1. 先在完整 DSR-sub 数据集（1209 样本）上训练 500 步
2. 记录每个样本在不同 checkpoint 的准确率 $s_i$
3. 计算方差，从高到低排序——**高方差 = 模型"犹豫"的样本 = 信息量更大**

#### 训练目标（GRPO）

三个 loss 组件协同：
- **Policy Gradient Loss**（GRPO-style）：主驱动
- **KL Divergence Loss**（β=0.001）：防止语言质量坍缩
- **Entropy Loss**（α=-0.001，最大化熵）：**关键——鼓励输出多样性**

#### 单样本训练的实际操作
- 单个样本复制填充训练 batch (size=128)
- 每 prompt 采样 8 个回答 → 每 rollout 步 8 次梯度更新

### 8.3 后饱和泛化（Post-Saturation Generalization）

这是论文最有趣的现象：

1. **训练准确率在 ~200 步就饱和了**（对单个样本 near-100%）
2. **但测试准确率继续提升**——持续数百步
3. 与 grokking 不同：移除正则化也不消失，主驱动是**Policy Gradient Loss 本身**

**熵的作用**：如果没有熵损失，训练准确率 100% 后梯度消失。适当调熵损失系数，模型保持略低于 100% → 保留有效梯度信号。

**彩蛋**：仅用熵损失（无准确率奖励）也能提升 MATH500 27.4%！

### 8.4 个人评注

> 1-Shot RLVR 对教育场景的含义极为深刻：**一个精心挑选的题目 + RL 可能就足够了**。加上后饱和泛化现象，这暗示 RLVR 的核心机制可能是"让模型多思考几次同一个问题，从中学会通用的思维习惯"，而非"从海量题目中学到解题模板"。

---

## 9. GRPO-LEAD: Difficulty-Aware RL for Concise Mathematical Reasoning

- **发表**: arXiv: 2504.09696 (2025)

### 9.1 三个改进

1. **长度依赖的准确率奖励**：鼓励简洁推理——长而不准 → 惩罚更大
2. **错误答案显式惩罚**：与 Negative RL 的思路一致
3. **难度感知的 advantage 重加权**：难题上对/错的信息量 > 简单题

### 9.2 个人评注

> 解决 GRPO 的冗长问题和稀疏奖励问题。难度感知的 advantage 重加权与 Education RL（UCO/Scaffold Reward）的思路一脉相承，都试图让算法关注"恰好有点难"的区域。

---

## 10. Self-Evolving Curriculum for LLM Reasoning

- **作者**: Xiaoyin Chen 等 (Mila / Bengio)
- **发表**: arXiv: 2505.14970 (2025.05)

### 10.1 方法

将 RL 微调中的课程选择建模为**非稳态多臂老虎机**：

- **每只臂** = 一个问题类别（难度级别/题目类型）
- **非稳态** = 随着 LLM 策略更新，每只臂的期望奖励分布会变化
- **奖励信号** = **绝对 advantage** $|Â_t|$ ——恰好难度适中的题目给出最大梯度
- **更新** = TD(0) 时序差分：$Q_{t+1}(c) = α·r_t(c) + (1-α)·Q_t(c)$

### 10.2 结果

| 领域 | 相比随机课程提升 |
|------|:---:|
| Countdown（规划） | +13% |
| Zebra Puzzles（规划） | +21% |
| ARC-1D（归纳推理） | +22% |
| AIME24（数学） | **+33%** |

### 10.3 个人评注

> 课程学习的自动化一直是个难问题。这篇用 bandit + advantage 信号的方法非常优雅。教育场景中，"什么难度的题目该在什么时候给学生"本来就是核心问题——这可能是将 RL 课程学习应用于教育的最直接路径。

---

# 第三部分：奖励设计

---

## 11. Reward Granularity in RLVR: Process vs. Outcome Rewards

- **作者**: Anagha Radhakrishna Palandye 等
- **发表**: arXiv: 2607.02869 (2026.07)

### 11.1 实验设计

Qwen2.5-0.5B + GRPO + GSM8K，对比五种奖励配置：

| 奖励条件 | GSM8K 测试准确率 |
|----------|:--------------:|
| **纯过程级** | **63.73%** |
| 纯结果级 | 53.75% |
| 混合（λ=0.9 过程） | 60%+ |
| 混合（λ=0.5） | ~57% |
| 混合（λ=0.1 过程） | **低于纯结果级** |

### 11.2 关键分析

- 过程模型：**结构不一致但算术正确**的推理链
- 结果模型：**简洁但推导易错**的推理链
- 混合权重不当（λ=0.1）会**比纯结果更差**——冲突的优化信号适得其反

### 11.3 个人评注

> 过程级奖励优于结果级是直觉上合理的，但这里关键的二阶发现是：**混合权重调不好反而有害**。对于教育应用，如果要用过程奖励来优化教学行为（如检查学生每一步是否有概念错误），这个"粒度设计是一阶决策"的结论非常重要。

---

## 12. CORE: Concept-Oriented Reinforcement

- **作者**: Zijun Gao 等 (UIUC / ASU)
- **发表**: arXiv: 2512.18857 (2025.12)

### 12.1 核心问题：定义-应用鸿沟

LLM 能完美背诵数学定义，但解题时无法正确应用。标准 RLVR 只优化终端正确性，不够细粒度来教模型"哪个概念、在哪里用、怎么用"。

### 12.2 方法

#### 数据构建
- 来源：《高等代数》（第三版）中译英，**避免预训练污染**
- 产出：236 个概念文本 + 703 个例题 + 140 个选择题

#### 概念诊断
- GPT-4o 生成 1200 道选择题
- **鲁棒评估协议**：每道题需要同时答对原始版本和 3 个选项置换版本才算通过
- 结果：Qwen-2-Math-7B 从 74.33% 跌到 45.92% → 暴露表面启发式

#### 三种 CORE 变体（基于 GRPO）

| 变体 | 机制 |
|------|------|
| **CORE-Base** | 直接在概念对齐的测验数据集上 GRPO |
| **CORE-CR** (Trajectory Replacement) | 当所有采样轨迹都错 → 注入概念片段 → 重新生成概念引导轨迹 → 替换失败轨迹 + 奖励加成 |
| **CORE-KL** (KL Alignment) | 同上触发 + 添加 forward-KL loss，鼓励标准策略对齐概念引导分布 |

### 12.3 结果（Qwen2-Math-7B）

| 方法 | Textbook | TheoremQA | MATH | GSM8K |
|------|:------:|:------:|:------:|:------:|
| Vanilla | 46.4% | 34.6% | 55.3% | 89.8% |
| CORE-Base | 50.7% | 40.4% | 59.5% | 90.8% |
| CORE-CR | 52.1% | 42.3% | 58.4% | 91.1% |
| **CORE-KL** | **55.7%** | **44.2%** | **59.5%** | 90.7% |

**关键**：52.6% 的 CORE 成功案例是纯粹的**概念选择**问题（模型显式调用并正确应用了目标数学概念）。

### 12.4 个人评注

> CORE 直接触及了教育中最核心的问题：**从"能背诵"到"能应用"**。概念片段（concept snippets）仅注入训练，推断时不需要——这类似于老师讲课时提醒概念，考试时不提醒。对于教育解题应用，CORE 的思路（用概念引导 RL 训练）比纯粹的结果/过程奖励更高一个层次。

---

## 13. Forge: Quality-Aware RL & Rewarding the Rare

### 13.1 Forge：超越二元奖励

- **发表**: ACL 2026 Findings
- **核心**: 用**质量感知奖励**（而非对/错二元）训练 NP-Hard 优化
- **结果**: 相比二元奖励提升 28.8%，迁移到数学 +2.2%

### 13.2 Rewarding the Rare：奖励稀有策略

- **发表**: ACL 2026 Findings
- **核心**: 按**策略聚类**，按簇大小反向加权——小簇（稀有策略）权重高
- **效果**: 防止探索坍缩，提升 pass@k 和 AUC

### 13.3 个人评注

> 两篇都指向同一个方向：**奖励信号的信息密度**。二元奖励浪费了大量信息——是"恰好对了"还是"擦边对了"？是"常规解法"还是"创意解法"？教育场景同理：学生答案的评分不应该只是对/错。

---

# 第四部分：教育应用

---

## 14. UCO: Multi-Turn Interactive RL for Adaptive Teaching

- **作者**: Shouang Wei 等 (华东师范大学 / 浙江大学 / CUHK-Shenzhen)
- **发表**: arXiv: 2511.08873 (2025.11 / 2026.01)
- **代码**: github.com/Mind-Lab-ECNU/UCO

### 14.1 核心创新：两个新奖励函数

#### Progress Reward（进步奖励）

**理论**: 认知进步 = 信息论熵减——学生从高熵（不确定）→ 低熵（确定）。

两个代理指标：

**① Potential Capability Score**：用 Oracle 模型（Gemini-2.5-Pro）生成 N 个正确候选答案，计算学生模型对这些候选的 log-probability——学生越"接近"正确理解，分数越高

**② Semantic Quality Score**：用 BGE embedding 计算学生实际输出与候选答案的语义相似度

#### Scaffold Reward（支架奖励）

**理论**: 维果茨基的最近发展区（ZPD）。

**五级支架层次**：

| 级别 | 类型 | 例子 |
|:----:|------|------|
| ℓ₀ | 元认知提示 | "你能想想我们之前学过的哪个定理可以用？" |
| ℓ₁ | 策略提示 | "可以尝试从条件出发推导" |
| ℓ₂ | 概念提示 | "需要用勾股定理" |
| ℓ₃ | 分步提示 | "第一步...第二步..." |
| ℓ₄ | 示例演示 | 给出完整类似题解答 |

**ZPD 定位**: 找到让学生成功率最高的支架级别 → **降一级** → 进入 ZPD

**奖励**: 命中 ZPD 正奖励 / 偏离 ZPD 负惩罚

### 14.2 训练设置

- 教师模型：Qwen2.5-7B-Instruct（可训练）
- 学生模型：Qwen2.5-14B-Instruct（冻结）
- Oracle：Gemini-2.5-Pro
- 每次采样 G=4 条交互 rollout
- GRPO 优化，累积折扣奖励 + 组内 advantage 归一化

### 14.3 关键结果（BigMath 基准，对比 11 个基线）

| 模型 | Δ Solve Rate ↑ | Leak Solution ↓ | Ped-RM ↑ |
|------|:---:|:---:|:---:|
| Qwen2.5-7B-Instruct（基座） | 11.3 | 29.3 | -0.2 |
| GPT-4o | 33.1 | 35.2 | 1.5 |
| DeepSeek-V3 (671B) | 39.3 | 46.6 | -1.5 |
| **UCO (7B)** | **30.2** | **12.9** | **4.6** |

> **仅 7B 参数的 UCO 在教学效果、答案泄露控制、教学品质三个维度实现协同最优。答案泄露率仅 12.9%，远低于 DeepSeek-V3 的 46.6%！**

### 14.4 消融实验

| 配置 | Δ Solve Rate | Leak Solution |
|------|:---:|:---:|
| UCO 完整 | 30.2 | 12.9 |
| -Scaffold Reward | 23.5 (**-6.7**) | 20.2 (**+7.3**) |
| -Progress Reward | 28.6 (-1.6) | 15.6 (+2.7) |

**Scaffold Reward 贡献最大**——ZPD 导向的支架调节对自适应教学至关重要。

### 14.5 个人评注

> UCO 是我在 2025-2026 教育 RL 领域看到的最精心设计的框架。它把教育心理学的核心概念（ZPD、支架式教学）**直接编码进奖励函数**，并且用"单向优化"（仅优化教师模型）避开了共适应问题。答案泄露率从 DeepSeek-V3 的 46.6% 降到 12.9%，彻底解决了 LLM 当老师时"忍不住直接给答案"的顽疾。

---

## 15. PedagogicalRL-Thinking: 奖励模型的"教学思维"

- **作者**: Unggi Lee 等
- **发表**: arXiv: 2601.14560 (2026.01)

### 15.1 两项创新

**① Pedagogical Reasoning Prompting**
- 引导模型的内部推理使用 **Polya 四步法**（理解问题→制定计划→执行计划→回顾验证），而非通用指令
- 这比"请做一个好老师"具体得多

**② Thinking Reward**
- 显式评估 `<think>` 标签内部的**推理轨迹的教学质量**
- LLM Judge (GPT-4o-mini) 评分：教学适当性、对学生的理解程度、元认知意识

### 15.2 实验设计（5 种条件）

| 条件 | 基座模型 | Thinking | Prompt | Think Reward |
|------|---------|:------:|--------|:---:|
| NoThink | Qwen2.5-7B | ✗ | Normal | ✗ |
| Think NoReward | DeepSeek-R1-Qwen3-8B | ✓ | Normal | ✗ |
| Think Reward | DeepSeek-R1-Qwen3-8B | ✓ | Normal | ✓ |
| Ped. Think NoReward | DeepSeek-R1-Qwen3-8B | ✓ | Pedagogical | ✗ |
| **Ped. Think Reward** | DeepSeek-R1-Qwen3-8B | ✓ | Pedagogical | ✓ |

### 15.3 结果

| 条件 | Δ Solve Rate | Leak Rate ↓ | Helpful Rate ↑ |
|------|:---:|:---:|:---:|
| NoThink | 0.120 | 0.300 | 0.180 |
| Think NoReward | 0.281 | 0.180 | 0.730 |
| **Ped. Think Reward** | **0.294** | **0.172** | **0.776** |

**关键发现**：
- 启用 thinking 本身就已经大幅提升辅导质量（0.120→0.281）
- 领域特定的教育理论提示**优于通用提示**
- Thinking Reward 在与 Pedagogical Prompting 结合时**效果最佳**——两者有协同效应
- 仅在数学辅导数据上训练的模型**迁移**到了未见过教育基准

### 15.4 行为层面变化
- 代码本分析显示：奖励教学思维后 → 更结构化的教学决策
- 步骤引导增加（18.22% vs 16.73%）
- 过度表扬减少
- 更多"响应式教学"模式

### 15.5 个人评注

> 这篇论文的独特价值在于它**同时优化了"教学输出"和"教学思维"**。Thinking Reward 是教育 RL 的一个新维度——我们不仅要模型教得好，还要它"想得对"。Polya 框架的选择很巧妙，因为它本身就是一个已被验证有效的元认知框架。

---

## 16. MathTutorBench + RL Alignment Framework for LLM Tutors

- **作者**: Jakub Macina 等 (ETH Zurich)
- **发表**: EMNLP 2025 Main (两篇：基准 Oral + RL 框架)
- **代码**: github.com/eth-lre/PedagogicalRL

### 16.1 MathTutorBench：首个全面的辅导评估基准

评估导师模型的**三项核心技能**：

| 维度 | 测量内容 |
|------|---------|
| **Math Expertise** | 解题能力、生成 Socratic 子问题 |
| **Student Understanding** | 判断学生正确性、定位多步解答中的错误 |
| **Pedagogical Ability** | 支架生成、教学指令遵循 |

**关键创新: Scaffolding Score**
- 专门训练的奖励模型（人工标注的教学偏好数据）
- 二元排名损失 + 教学标准满足的 margin
- 区分专家 vs 新手教师：**0.84 准确率**，远超 LLM-as-a-judge

### 16.2 RL Alignment Framework：从"答题"到"教题"

#### MDP 建模

- **状态**: 完整对话历史
- **动作**: 导师下一句
- **转移**: 导师回合 → 采样自 $\pi_\theta$；学生回合 → 采样自冻结学生 LLM
- **关键**: 在策略（on-policy）——模型基于自己的交互训练，避免分布偏移

#### 双奖励设计

1. **Post-dialog Solve Rate** ($r_{sol}$): 教学结束后学生独立解题的正确率
2. **Pedagogical Quality** ($r_{ped}$): LLM judge 评估（答案为泄露？有帮助吗？）——**两个 judge 都通过才算通过**

$$r = r_{sol} + (r_{ped} - 1) \cdot \lambda$$

- $\lambda$ = 可控惩罚超参数 → 在**Pareto 前沿**上导航

### 16.3 关键结果

- **7B RL 训练模型几乎匹敌闭源 LearnLM**，无需人类标注
- 高 $\lambda$ → 答案泄露 ~5%，同时保持有意义的 Δ Solve Rate
- 与 SFT 不同，RL 训练的模型**不损失**通用推理能力（MMLU/GSM8K 保持）

### 16.4 个人评注

> MathTutorBench 和 PedagogicalRL 是教育 RL 领域最完整的基础设施。Scaffolding Score（区分专家/新手教师达到 0.84）意味着我们有了可扩展的自动化教学评估。Pareto 前沿设计让人能按需选择"效果 vs 不泄答案"的折中点——教育场景下这个 tradeoff 非常重要。

---

## 17. TEI: Tutoring Effectiveness Index

- **作者**: Shim Jaechang & Unggi Lee
- **发表**: CIKM 2026, arXiv: 2605.30666

### 17.1 核心创新：无需训练、无需 Judge 的四信号指数

从**冻结模型**的多个候选回复中选出最佳教学回复，仅用文本操作。

**四个信号**：

| 信号 | 来源 | 测量内容 |
|------|------|---------|
| **V — Verify Ratio** | Regex 匹配 thinking trace | 模型自我验证频率（Schoenfeld Verify 阶段） |
| **M̃ — Math-Step Density** | Regex 匹配可见输出 | 数学步骤指导密度 |
| **Q — Ends-Question Rate** | Regex 匹配可见输出 | 反问率悖论（过多 = Socratic over-pattern） |
| **D — Deep-Reasoning Gate** | DTR probe（forward hooks） | 跨层深度推理的二元 bonus |

$$TEI = V + 0.75 \cdot M̃ - Q + 0.5 \cdot \mathbb{1}[DTR \ge 0.4]$$

### 17.2 TEI@8 效果

- 预错误场景改进率：59% → **81.9%**（冻结 DeepSeek-R1-8B）
- Token 成本：16,334（Cons@8 的一半）

### 17.3 教育 RL 的 Alignment Tax 警示

**令人警醒的发现**：对基座模型做教学 GRPO 对齐后：

| 指标 | 对齐前 | 对齐后 |
|------|:---:|:---:|
| Δ Solve Rate | **+0.180** | **-0.012**（不如没有老师！） |
| Thinking 长度 | 1,764 | 119 (**-93%**) |
| 内容知识 | 2.18 | 0.62 (**-71%**) |
| 教学知识 | 2.20 | 0.44 (**-80%**) |
| Helpfulness | 0.220 | 0.416 (**↑ 但无学习效果**) |

**RL 对齐让模型变得更"乐于助人"（给答案），但彻底摧毁了学生的学习效果！**

### 17.4 个人评注

> TEI 是教育 RL 领域的一个重要"清醒剂"。Alignment Tax 的发现警告我们：**如果奖励函数只关注表面教学指标（如 helpfulness），RL 会把模型训练成"直接给答案的讨好型老师"，这恰好是教育中最糟糕的。** 这也解释了为什么 UCO 的 Scaffold Reward（抑制直接给答案）如此重要。

---

## 18. Stanford + 台北市 RCT：真实课堂 RL 辅导系统

- **作者**: Chung, Zhang, Kung, Bastani & Bastani (Stanford)
- **发表**: 2025

### 18.1 部署

- **台北市政府 + 10 所高中**，Python 编程教学
- GenAI 聊天机器人 + **RL 算法为练习题排序**
- 信号来自学生-聊天机器人交互

### 18.2 RCT 结果

- 自适应排序提升独立期末考试 **+0.15 SD**（≈ **6-9 个月学龄**）
- 实验组学生编程能力显著提升

### 18.3 个人评注

> 这是目前教育 AI RL 领域最具说服力的真实课堂证据之一。+0.15 SD 在社会科学中是一个相当可观的教育干预效应量。它将 RL-for-education 从实验室带到了真实课堂，证明了这条路能落地。

---

## 19. History-Aware Profiles for Student Simulation

- **作者**: Zhangqi Duan, Andrew Lan 等 (UMass Amherst + Eedi)
- **发表**: arXiv: 2605.30051 (2026.05)

### 19.1 方法：三阶段框架

**Stage 1: Profile Generator**
- 将学生学习历史压缩为**五维度结构化档案**：
  1. 知识状态（每个知识点的掌握率）
  2. 知识获取（趋势）
  3. 错误概念（重复犯错）
  4. 对话行为（常见对话动作）
  5. 语言风格

**Stage 2: Student Simulator**
- 给定问题+档案+对话历史，预测学生回答
- SFT + Profile-Aligned DPO：保证模拟器忠实于分配到的档案

**Stage 3: Profile Generator Tuning via GRPO**
- 冻结模拟器 → 用 GRPO 优化档案生成器
- 奖励：模拟器对真实对话的 log-likelihood

### 19.2 关键贡献

- QA 历史 + 对话历史**互补**：前者揭示知识，后者揭示行为和细粒度认知
- RL 训练显著提升模拟保真度（超越 prompting-only 和 SFT-only）

### 19.3 个人评注

> 学生模拟器是教育 AI 的关键基础设施——它让我们能以极低成本大规模评估和训练 AI 导师。用 GRPO 来优化档案生成器（而非直接优化对话）是一个聪明的解耦：先建模学生"是什么样的人"，再让这个"人"去说话。

---

# 第五部分：跨领域迁移与扩展

---

## 20. Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math

- **作者**: Bo Pang 等 (Salesforce AI Research / UCLA)
- **发表**: arXiv: 2510.26143 (2025.10)

### 20.1 核心策略

**两阶段课程**：

| 阶段 | 内容 | 目的 |
|------|------|------|
| Stage 1 | 冷启动 SFT（10K 数学 + R1 推理轨迹）→ **纯数学 RL** (DAPO/GRPO) | 在预训练优势领域引出核心推理技能 |
| Stage 2 | **联合 RL**：数学+STEM+代码+模拟+逻辑+表格 | 跨领域迁移和精炼 |

**核心假设: 数学是 RL 推理的"健身房"**——数学 RL 容易做（可验证），且引出的认知技能（子目标设定、枚举、回溯、验证）被证明是跨领域可迁移的。

### 20.2 认知技能追踪

| 技能 | 联合 RL | 直接 RL | 数学优先 RL |
|------|:---:|:---:|:---:|
| 子目标设定 | 高 | 中 | 高 |
| 枚举 | 中 | 低 | 中 |
| **回溯** | 低 | **几乎为零** | **显著** |
| **验证** | 低 | **几乎为零** | **显著** |

**数学优先引出后**，回溯和验证在 Stage 2 联合 RL 中**自动出现在非数学领域**。

### 20.3 结果（Qwen3-4B vs 32B 基线）

4B 模型在多数基准上**匹敌或超越** 32B 模型：
- GPQA（STEM）：4B 53.16 vs 32B 50.63
- HumanEval（代码）：4B 90.85 = 32B 90.85
- BoxNet（逻辑）：4B **93.80** vs 32B 0.12（差距巨大）

### 20.4 个人评注

> 这项工作的实践指导意义极大：如果你想让模型在多个领域都会推理，**先从数学 RL 做起**，然后联合训练。认知技能分析（回溯和验证仅在数学优先训练后出现）也为"数学训练为什么能促进通用推理"提供了机制性解释。

---

## 21. Crossing the Reward Bridge: Expanding RLVR Across Domains

- **作者**: Yi Su 等 (Tencent AI Lab / 苏州大学)
- **发表**: arXiv: 2503.23829 (2025.03)

### 21.1 核心问题

RLVR 在数学和代码领域成功，但**无法扩展到医学、化学、心理学、经济学、教育**——这些领域的答案是非结构化的，没有简单的对/错验证。

### 21.2 方法：生成式软奖励模型

**三步流程**：

1. **探索数据生成**：大 Teacher（Qwen2.5-72B）对模型输出做二元判断
2. **奖励模型蒸馏**：将 Teacher 的判断能力蒸馏到 **7B 学生模型**（判断比生成简单）
3. **最终策略训练**：7B 蒸馏模型作为 GenRM，提供**软奖励**（正确概率 0.8，而非 0/1）

**软奖励是核心创新**：医学/教育中答案可以是部分正确的（"诊断对了但治疗方案不够好"=0.7 而非 0）。

### 21.3 关键结果

- 7B 基座 + RLVR **超越** Qwen2.5-72B-Instruct 和 DeepSeek-R1-Distill-Qwen-32B（最高 **+8%**）
- 一个跨领域 7B Judge 适用于所有科目
- 开源 57 万条多领域训练数据

### 21.4 个人评注

> 这篇为 RLVR 打开了一扇通向非封闭领域的大门。在教育和医学场景，答案"好坏"是连续的（而非二元的），软奖励模型提供了一种优雅的解决方案。但我担忧的是：蒸馏的 7B Judge 会继承 Teacher 的偏见，而教育/医学领域的偏见可能很隐蔽且高度敏感。

---

## 22. QuestA: Question Augmentation for RL Reasoning

- **作者**: Jiazheng Li 等 (清华 / Amazon / Stanford)
- **发表**: ICLR 2026

### 22.1 核心思想

RL 的困境：
- 简单题训练 → Pass@1 提升但 Pass@k **退化**（熵坍缩）
- 难题训练 → 基座成功率接近零 → 奖励稀疏 → 无法学习

**QuestA 方案**：给难题**预置部分解**（如 50% 或 25% 的解题步骤）作为 "Hint"，给 RL 提供更密集的奖励信号。

**课程**：p=50% hint → p=25% → 最终无 hint

### 22.2 结果（1.5B 参数）

| 基准 | QuestA | 提升 |
|------|:------:|:---:|
| AIME 2024 | **72.50%** | **+10.73%** |
| AIME 2025 | **62.29%** | **+12.79%** |
| HMMT 2025 | **41.67%** | **+10.11%** |

**同时提升 Pass@1 和 Pass@k**——解决了 RL 对多样性的损害。

### 22.3 个人评注

> QuestA 的部分解策略是教育场景中最自然的类比：**老师给 hint**。它的优雅之处在于简化——不需要复杂的过程奖励模型，仅用数据层面（partial solution prepended）的技巧就解决了难题上 RL 的冷启动问题。教育应用中，这可以发展为"自适应 hint"系统。

---

## 23. SATURN: SAT-Based RL to Unleash LLM Reasoning

- **作者**: Huanyu Liu 等 (北京大学)
- **发表**: NeurIPS 2025 **Spotlight**

### 23.1 为什么用 SAT 问题？

| 需求 | SAT 如何满足 |
|------|------------|
| **可扩展性** | 程序化生成，无需人工标注 |
| **可验证性** | 代入变量赋值即可验证 |
| **难度可控** | 调节变量数 k、子句数 l、每个子句的文字数 n |

### 23.2 双环课程

| 循环 | 角色 | 行为 |
|------|------|------|
| 课程估计（Teacher） | 评估 pass@1 | 准确率 > ε → 增加难度；否则触发训练 |
| LLM 训练（Student） | GRPO RL | 在当前难度生成 SAT 问题，训练至超越阈值 |

### 23.3 结果

| 基准 | SATURN-1.5B | SATURN-7B |
|------|:---:|:---:|
| SAT 平均 pass@3 提升 | **+14.0** | **+28.1** |
| AIME 24/25 | **28.3** (+6.7) | 48.3 (-1.7) |
| AMC 22/23 | **73.5** (+8.4) | **85.5** (+4.8) |

**关键**: 与 SFT 不同（SFT 提升数学但**降低** LiveCodeBench），SATURN**全面提升**所有基准。

### 23.4 自我验证行为

训练过程中观察到模型自发学会**验证中间结论**——因为 SAT 的每个子句都必须检查是否满足。这个习惯迁移到了数学和编程。

### 23.5 个人评注

> SATURN 最漂亮的洞察在于**选择了 SAT 作为训练环境**。SAT 天然是一个"推理 gym"——可调难度、自动验证、无限生成。对于教育场景的启示：也许我们应该用**可程序化生成的逻辑推理题**（而非昂贵的人工标注数学题）来做 RL 训练的基础阶段。

---

## 24. SwS: Self-aware Weakness-driven Problem Synthesis

- **发表**: NeurIPS 2025

### 24.1 方法

1. 从 RL 训练失败中**识别模型弱点**
2. 提取**核心概念**
3. **合成新题**精准针对那些弱点
4. 7B: +10%，32B: +7.7%，跨 8 个推理基准

### 24.2 个人评注

> 与 UCO 的 ZPD 发现和 Self-Evolving Curriculum 的 bandit 选择异曲同工。"哪里不会练哪里"——这对教育 AI 是最基本也是最重要的能力。

---

# 第六部分：综合评注与方向判断

## 跨论文主题分析

### 1. RLVR 的"暗面"——我们真的在"学习"吗？

Spurious Rewards、Negative RL、TEI 的 Alignment Tax 三篇论文放在一起，揭示了一个令人不安的模式：

- **Spurious Rewards**: 随机奖励也能在 Qwen 上提升 21% → GRPO clipping 在"挤压"预训练先验
- **Negative RL**: 压制错误路径比奖励正确路径更重要 → RL 更多是"排除法"而非"学会新知识"
- **TEI Alignment Tax**: 用 RL 对齐"好教学" → 模型学会了给答案（高 helpfulness），但彻底摧毁了学习效果（负的 Solve Rate）

**综合判断**: RLVR 目前可能在**精炼预训练知识**而非**发现新推理能力**。对教育解题来说，这意味着 RLVR 适合用来"激发基座模型已有的教学潜力"，但不适合用来"让模型学会它完全不会的推理模式"。

### 2. 奖励设计的进化路径

```
二元对/错 → 过程级 → 质量感知 → 软概率 → 教学理论编码
  (R1)    (Reward Granularity)  (Forge)  (Cross Bridge)  (UCO/PedagogicalRL)
```

我认为教育 RL 的下一步应该走向**多维度奖励矩阵**：
- 学生进步（Progress Reward）
- 教学策略适当性（Scaffold Reward）  
- 答案泄露控制（Leak Penalty）
- 教学思维质量（Thinking Reward）
- 概念理解深度（CORE-style concept probes）

### 3. 从数学 RL 到教学 RL 的迁移路径

目前最清晰的路线图：

1. **数学 RL 打下推理基础**（Reasoning Curriculum 证明了数学→多领域的迁移）
2. **引入教育理论编码的奖励函数**（UCO 的 ZPD + Scaffold, PedagogicalRL 的 Polya + Think Reward）
3. **多轮交互 RL**（UCO, Macina et al. 的 on-policy RL）
4. **用模拟学生降低评估成本**（History-Aware Profiles）
5. **用 TEI 等无需训练的指标做部署前筛选**（避免 Alignment Tax）

### 4. 最值得关注的方向（个人排序）

| 优先级 | 方向 | 理由 |
|:---:|------|------|
| 1 | **教学中 RL 的 Alignment Tax 问题** | TEI 的结果太惊人了——把好老师训练成坏老师只需一次 GRPO |
| 2 | **过程奖励模型（PRM）的自动化** | 标注成本仍是瓶颈，隐式 PRM (FreePRM/SelfPRM) 方向值得 follow |
| 3 | **RL for 概念理解（非仅解题）** | CORE 揭示了"能背诵≠能应用"的鸿沟，这是教育的本质问题 |
| 4 | **多轮交互 RL + 真实课堂验证** | 台北 RCT 是第一步，需要更多真实世界证据 |
| 5 | **Spurious Rewards 的深层机制** | 如果 RLVR 大部分是"挤压预训练"，那 RL 路线需要根本性反思 |

### 5. 开源生态

| 工具 | 用途 | 成熟度 |
|------|------|:---:|
| verl | GRPO/DAPO/PPO 训练框架 | 🟢 |
| OpenR | PRM + RL + 搜索统一框架 | 🟢 |
| rStar | MCTS + 自进化 | 🟢 |
| TRL | HuggingFace RL 训练 | 🟢 |
| PedagogicalRL | 教学 RL 专用框架 | 🟡 |

---

## 参考文献

（详见调研总笔记 `rl-math-reasoning-survey-2025-2026.md` 的参考文献列表）
