---
title: "Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning"
method_name: "Agon"
authors: [Vladislav Beliaev]
year: 2026
venue: arXiv
tags: [rl, grpo, reasoning, multi-model, competitive-rl, self-play, draft-and-challenge]
zotero_collection: 
image_source: online
arxiv_html: https://arxiv.org/html/2607.07690
arxiv_pdf: https://arxiv.org/pdf/2607.07690v1
created: 2026-07-09
---

# 论文笔记：Agon

## 元信息

- **标题**: Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning
- **作者**: Vladislav Beliaev (Independent Researcher, thinkdense.ai)
- **年份**: 2026 (提交于 2026-07-08)
- **发表**: arXiv 预印本 (2607.07690v1)
- **学科领域**: cs.LG, cs.AI, cs.CL
- **页数**: 15 pages, 7 figures, 8 tables
- **代码/资源**: 未公开 (使用 TRL + vLLM 实现)

## 一句话总结

> Agon 让两个能力相当但行为不同的模型在 RL 训练中互为隐式评分器 —— 一个起草解答，另一个阅读后挑战；通过奖励"击败对手"来隐式评判推理质量，无需 process reward model，无需人工标注。在 DeepMath 困难集上，Agon 将 GRPO 的 pass@1 翻倍，涨幅是未训练 Mixture-of-Agents 的约 8 倍。

## 核心贡献

1. **将推理质量重新定义为隐式竞争奖励**：用两个竞争模型互为评分器，把"什么是好的推理"这个无标注问题转化成了一个 head-to-head 博弈，由训练过程中的竞争结果来隐式评判推理质量。
2. **提出 Agon 训练框架**：通过 draft-and-challenge rollout、逐步角色轮换、竞争性奖励（正确性 + 转换奖励 conversion bonus）联合训练两个不同策略。运行在标准 GRPO 训练器上，核心优化器无需改动。
3. **设计 2x2 消融矩阵**（竞争 x 信息交换），加上 self-refinement 和 zero-training 控制组，系统性地隔离竞争与交换各自的贡献。
4. **实验覆盖多个模型家族和领域**：Qwen3 (0.6B/1.7B/4B)、Qwen3.5、Gemma 4；DeepMath + CodeContests。

## 问题背景

### 要解决的问题 (GRPO only grades final answer, not reasoning trace)

GRPO 等基于可验证奖励的 RL 方法只评价最终答案，不评价推理过程本身。在难题上，这会产生一个强烈的偏差：模型学会"写更多"而不是"思考更好" —— 更长的推理链意味着更多碰巧踩中正确答案的机会，所以增加文本量是提高期望奖励最廉价的方式。推理链充斥着 hedging 和 backtracking（"hmm," "wait," "let me reconsider"），准确率增长远慢于长度增长。

**直接后果**: 模型产生更多的推理量，但每个 token 的信号密度并未提高。

### 现有方法的局限

| 方法 | 局限 |
|------|------|
| Length penalty / length-controlled objectives (Aggarwal & Welleck, 2025) | 作用于症状（长度），而非根因（推理质量无信号） |
| Dynamic sampling, discard all-correct/all-wrong groups (DAPO; Yu et al., 2025) | 同样是症状层面的补救 |
| Process reward models (Lightman et al., 2023) | 昂贵、脆弱、本身不可验证；步骤级标注不存在 |
| Self-play (SPIN, Absolute Zero, R-Zero) | 单一策略在自己的信号上优化，会强化导致错误的盲点，自我纠正不可靠 (Huang et al., 2024) |
| Multi-agent debate / MoA (Du et al., 2023; Wang et al., 2024) | 合作式聚合，趋向共识，当模型能力不同时稀释质量；均为推理时冻结模型的操作 |

### 本文的动机

核心问题：**能不能让另一个模型来提供当前模型缺失的推理质量信号？**

关键洞察：如果一个不同的策略尝试同一个问题，我们奖励每个模型"超越对手"，那么推理链就被对手**隐式评分**了 —— 导致胜利的步骤被强化，被对手利用的填充内容被惩罚。这不需要任何 process label。

## 方法详解

### Draft-and-Challenge protocol

每一步（Figure 2, Algorithm 1）指定一个 **drafter（起草者）** 和一个 **challenger（挑战者）**：

1. **Drafter 阶段**: Drafter 从纯问题 prompt 生成 N 个 rollout {a_i}，接收普通 GRPO 更新（仅正确性 + 格式，无对手项）。这个 standalone stream 训练每个 adapter 独立解题的能力。
2. **Challenger 阶段**: Challenger 条件化于配对对手的解答摘要（summary）—— 即模型在私有推理块后写的答案部分，不包括原始推理链（原始思考 trace 被丢弃，因为 token 量大且充满探索噪声）。最终答案也被隐藏以防直接复制。Challenger 为每对生成一个 rollout b_i ~ π_chal(·|x, a_i)，共 N 个。
3. **奖励计算**: Challenger 接收竞争性梯度。每个 stream 的 GRPO advantage 在其自己的 N 个样本组内计算。

**角色轮换**：每一步后角色互换 —— 偶数步 A 起草 B 挑战，奇数步反之。不轮换时 pass@1 从 61 降到 52 (Table 7)。

### Why two models? (comparably strong, behaviorally different)

Post-training RL 本质上是自我改进：梯度从策略自身产生的 rollout 计算，因此会放大策略已有的行为模式，包括其系统性错误（盲点）。一个模型检查自己的工作继承了导致错误的偏见（为什么无辅助的自我纠正经常失败; Huang et al., 2024）。

**第二个模型在两个条件下才有效**：
1. **能力相当** (comparable strength)：否则弱方会模仿强方，博弈退化为蒸馏
2. **行为不同** (different failure modes/blind spots)：这样才能相互捕捉对方遗漏的错误

**本文的实现方式**：从同一 frozen base 上用两个不同的 LoRA adapter（rank-16, 仅 2% 参数开销），通过不同初始化（A 用标准零初始化，B 用零矩阵周围的小高斯噪声）和角色轮换导致的更新流差异来维持分歧。初始时两者仅因 adapter 噪声而不同，任何互补性必须从训练中涌现。

### Reward: correctness + conversion bonus

设 c(y) = r(x, y) ∈ {0, 1} 为正确性（verifier 输出），φ(·) 为格式奖励项。

**合作奖励 (coop)**：
```
R(b_i) = 2·c(b_i) + λ·φ(b_i)    (Equation 2)
```

**对抗奖励 (adv)**：
```
R(b_i) = 2·c(b_i) + c(b_i)·(1 - c(a_i)) + λ·φ(b_i)    (Equation 3)
                                       ↑ conversion bonus
```

其中 λ = 0.5（占正确性项的 25%），两个 stacks 相同以消除偏差。

**为什么 conversion bonus 的形式重要**：
- 简单的 margin `c(b_i) - c(a_i)` 效果差 —— 因为 c(a_i) 是 action-independent 的（由配对的 a_i 固定，不取决于挑战者的 rollout b_i），其减法项不贡献期望策略梯度，仅作为 per-sample baseline。
- Conversion bonus `c(b_i)·(1-c(a_i))` **乘以** action-dependent 的 c(b_i)，这种重新加权改变了梯度方向，并且由于组内对手上下文 {a_i} 的差异，将梯度倾斜向对手失败的样本。
- **关键要求**：组内对手难度必须有差异（per-rollout 不同对手），否则 conversion bonus 在 group-relative 标准化下贡献零梯度。共享对手消融（shared opponent, 32 pass@1）证实了这一点。

**奖励阶梯 (adv)**：R ∈ {0, 2, 3}。当对手失败时正确解题价值更高（3 vs 2）。

**可选的长度 tiebreak (Section 5.6)**：
```
R(b_i) += λ'·c(b_i)·c(a_i)·1[|b_i| < |a_i|]
```
仅在双方都正确且挑战者更短时触发，不会直接惩罚正确性。在 Qwen3-0.6B 上将 trace length 从 3.5k 压缩到 2.6k，准确率基本不变 (61 → 60)。

### Compute parity

每步 Agon 生成 N 个 drafter rollout + N 个 challenger rollout (2N)，而 vanilla GRPO 为 N。论文的不变量是 **generation budget**：每个方法每训练问题生成 2N 个 rollout，在相同的生成长度上限和相同的训练问题遍历次数下。Challenger 额外预填充对手摘要（短，远小于完整 trace，不进行等量化）。

**推理时**：两阶段级联 —— 一个 adapter 从纯 prompt 起草，另一个阅读草稿摘要并产出最终答案。这是两次顺序生成，相对于单次推理是双倍预算。因此 Table 3 包含了匹配预算的两阶段控制组（cross-refinement, self-refinement）。

### Inference cascade

推理时部署与训练相同：两阶段级联。两种级联方向（A→B 和 B→A）都被评估，报告在 held-out 集上事后选择的较好方向。

## 关键公式

### 1. GRPO 的 group-relative advantage

对于问题 x 和 verifier r(x, y) ∈ {0, 1}，GRPO 采样一组 G 个 rollout y_1, ..., y_G ~ π(·|x)，计算 group-relative advantage：

```
A_i = (r(x, y_i) - μ) / σ          (Equation 1)

其中:
μ = (1/G) · Σ_j r(x, y_j)          (组内均值)
σ = std_j r(x, y_j)                (组内标准差)
```

两组具有相同答案的 completion 获得相同的 advantage，**无论它们的推理过程如何**。零方差组（全正确或全错误）被丢弃，不产生训练信号。

### 2. 合作奖励（信息交换，无竞争）

```
R_coop(b_i) = 2·c(b_i) + λ·φ(b_i)     (Equation 2)
```

对手仅出现在上下文中，不在奖励中，隔离了信息交换的价值。

### 3. 对抗奖励（信息交换 + 竞争 = Agon）

```
R_adv(b_i) = 2·c(b_i) + c(b_i)·(1 - c(a_i)) + λ·φ(b_i)     (Equation 3)
```

Conversion bonus c(b_i)·(1-c(a_i)) 在对手失败的精确位置上将有效正确性权重从 2 提升到 3。

### 4. 可选的长度 tiebreak（辅助研究）

```
R(b_i) += λ'·c(b_i)·c(a_i)·1[|b_i| < |a_i|]
```

仅在双方都正确时打破平局，偏向更短的推理链。

## 关键图表

### Figure 1: 从答案评分到对手评分的转变

柱状图展示 DeepMath-hard held-out pass@1 (%)：
- Zero-shot: 23
- GRPO: 30 (+7)
- MoA (no train): 34 (+4 over GRPO)
- Coop (cooperative exchange): 46 (+16 over GRPO)
- Agon (competition + exchange): 61 (+31 over GRPO)

Agon 达到约 2x GRPO 和约 1.8x 未训练 MoA 的 pass@1，在相同的两次推理预算下。

### Figure 2: 一个 Agon 步骤的架构图

展示 drafter A 生成 N 个 rollout、接收 standalone GRPO 更新；challenger B 阅读每个对手的 solution summary（隐藏最终答案）、生成配对的 rollout、接收竞争性奖励。两个 adapter 每步都更新，角色每步轮换。

### Figure 3: 为什么需要第二个模型

- **左 (self-play)**: 单一策略在自己的 rollout 上优化，用自己的偏见重新审计 → 平台饱和
- **右 (Agon)**: 两个分歧策略，每个用不同的盲点评分对方，每个因获胜而被奖励，且因为两者都在优化，评分者与被评者共同进步

### Figure 4: 设计的分歧（目标状态示意）

- **左**: 两个同模型副本（不同温度）覆盖相同问题并在相同问题上失败 → 相关失败
- **右**: 能力相当但行为不同的两个模型有互补的错误分布 → 交叉评分有价值

### Figure 5: 窥视与超越（概念性示意图）

在一个轮换周期内聚合展示：挑战者阅读对手的 worked solution，捕捉其错误（如 eq. 3 中的符号错误），跳过其失败假设，借用其好的思路。两者都没有标注，但通过奖励超越对手的一方，隐式评分了推理质量。

### Figure 6: 示例交换

Drafter 在积分替换中掉了一个负号，导致 self-doubt 和大量重新探索（1.9k tokens）。Challenger 只看到 summary（不带答案的推导摘要），立即定位到符号错误，重新推导，在 4x 更少的 token 内得到正确答案。

### Figure 7: 准确率 vs 推理长度和训练曲线

- (a) Agon 位于左上角 —— 更高准确率、更短推理链（最终阶段 3.5k vs GRPO 的 8.1k）
- (b) 训练过程中 held-out pass@1 的演变。Self-refinement 在 GRPO 上方平台化，cooperative 更高，Agon (competition+exchange) 明显分离，且在预算结束时仍在改善

### Algorithm 1: Agon (一个优化器步骤)

```
1: input: batch of problems {x}; adapters A, B; group size N; step index t; mode ∈ {coop, adv}
2: (draft, chal) ← (A, B) if t even else (B, A)           // role rotation
3: for all x in batch do
4:     {a_i}^N_{i=1} ← π_draft(·|x)                        // standalone rollouts, plain prompt
5:     R(a_i) ← reward via Equation (2)                     // correctness + format, no opponent term
6:     {b_i}^N_{i=1} ← π_chal(·|x, a_i)                    // one per pair: challenger b_i sees a_i's summary
7:     R(b_i) ← reward via Equation (2)/Equation (3) by mode
8:     group-relative advantages of {R(a_i)} and of {R(b_i)} // Equation (1), per group
9: end for
10: GRPO update on π_draft (standalone stream) and on π_chal (competitive stream)
11: return updated adapters
```

## 实验

### Main results (DeepMath hard split, Qwen3)

**Table 3: 主要比较 (Qwen3-0.6B, DeepMath-hard held-out, %)**

| Method | pass@1 | avg. len (final stage) |
|--------|--------|------------------------|
| Zero-shot | 23 | 6.1k |
| Vanilla GRPO (baseline) | 30 | 8.1k |
| Self-refinement (control) | 32 | 7.9k |
| GRPO two-pass self-cascade (control) | 35 | 8.0k |
| MoA [Wang et al., 2024] (no train) | 34 | 6.9k |
| Competitive, shared opponent (single-pass) | 32 | 7.4k |
| Cooperative exchange | 46 | 5.1k |
| **Agon (competition + exchange)** | **61** | **3.5k** |

**Table 4: 95% Clopper-Pearson 置信区间 (300 held-out problems)**

| Method | pass@1 | 95% CI |
|--------|--------|--------|
| Zero-shot | 23 | [18.4, 28.3] |
| Vanilla GRPO | 30 | [25.0, 35.4] |
| Cooperative exchange | 46 | [40.3, 51.8] |
| **Agon** | **61** | **[55.2, 66.6]** |

所有增量均超过 CI 宽度（±5.5 pp）。

**Table 2: 迁移性检验 (GSM8K / MATH-500, Qwen3-0.6B)**

| Method | GSM8K | MATH-500 |
|--------|-------|----------|
| Zero-shot | 62 | 45 |
| Vanilla GRPO | 68 | 52 |
| Agon | 75 | 64 |

排序保持一致，绝对增益小于分布内（+13/+19 pp vs +38 pp），符合该方法压力针对 hard regime 的特点。

### Scaling & generality

**Table 5: 模型规模和外推到其他模型家族 (held-out pass@1, %)**

| Model | Zero-shot | Vanilla GRPO | Agon | Δ over GRPO |
|-------|-----------|-------------|------|-------------|
| Qwen3-0.6B | 23 | 30 | 61 | +31 |
| Qwen3-1.7B | 38 | 46 | 70 | +24 |
| Qwen3-4B | 52 | 59 | 71 | +12 |
| Qwen3.5-2B | 44 | 50 | 70 | +20 |
| Gemma-4-E4B | 50 | 58 | 73 | +15 |

关键结论：
- **小模型受益最大**（生活在更深的 hard regime 中），增益随基座增大而缩小但始终为正
- **Agon 0.6B (61) 超过 zero-shot Qwen3-4B (52) 和 vanilla GRPO on 4B (59)**，即一个 7x 更小的模型达到或超过更大模型
- 增益追踪的是 zero-shot 能力而非模型家族

**Table 6: 第二领域 —— CodeContests (easy) on Qwen3-1.7B**

| Method | pass@1 | avg. len (final stage) |
|--------|--------|------------------------|
| Zero-shot | 18 | 5.2k |
| Vanilla GRPO | 24 | 7.3k |
| Cooperative exchange | 29 | 4.8k |
| **Agon** | **34** | **3.9k** |

排序与 math 实验一致（exchange > baseline, competition > cooperation），traces 同样缩短。但绝对增益较小（单元测试奖励更稀疏，code traces 的对手预填充相对更长）。

### Ablations

**Table 7: 消融实验 (Qwen3-0.6B, held-out pass@1, %)**

| Factor | Variants (pass@1) |
|--------|-------------------|
| Competition (reward) | cooperative 46 / **adversarial 61** |
| Information exchange | shared opponent 32 / **per-rollout opponents 61** |
| Reward form | margin c(b_i)-c(a_i) 49 / **conversion bonus 61** |
| Role assignment | fixed roles 52 / **rotate 61** |

关键消融发现：
- **角色轮换至关重要**：固定角色下每个 adapter 只接收一种梯度，pass@1 降至 52
- **Conversion bonus > margin**: margin `c(b_i)-c(a_i)` (49) 几乎不优于 cooperative (46)，因为减法是 action-independent；conversion bonus 的乘法形式改变了梯度方向
- **Shared opponent (32)**: 组内对手难度无差异 → conversion bonus 在 group-relative 标准化下归零，理论预测与实验结果一致
- 两个杠杆贡献相当：信息交换 +16 pp (GRPO → coop)，竞争 +15 pp (coop → Agon)

**Drafter 单独评估**: 训练后的 drafter adapter 在独立解题（无对手摘要）时达到 46 pass@1（比 vanilla GRPO 高 +16 pp），尽管它只在半数步骤上进行 drafting（约 1/4 的 standalone rollout 梯度量）。假说是 challenger-stream 更新（验证和重新推导的技能）迁移到了独立解题。

### Analysis: why traces shorten

经验上，Agon 的 traces 比 GRPO 更短（Figure 7a）：挑战者阶段平均 3.5k tokens vs GRPO 的 8.1k。与未训练 MoA 的两阶段对照（6.9k）相比，Agon 的最终阶段仅为其一半长度。

- 主奖励函数**不含任何长度项**，缩短是**涌现的**
- 挑战者上下文中有候选解答 → 依赖长度的探索不再是找到正确答案的唯一路径
- 当对手解答正确时，挑战者 completion 比挑战者阶段均值短 35%（2.3k vs 3.5k）
- 模式（Figure 6）：挑战者定位到错误步骤，重新推导，然后停止

**可选密度杠杆（Table 8）**：

| Reward | pass@1 (%) | avg. len |
|--------|-----------|----------|
| Agon (headline, length-free) | 61 | 3.5k |
| + length tiebreak (λ') | 60 | 2.6k |

通过 tiebreak 可以主动控制长度，将涌现的缩短转化为直接的可调目标。

## 批判性思考

### 优点

1. **概念优雅**：将"推理质量信号"从一个需要标注的监督学习问题转化为一个博弈论问题 —— 两个对手的互动自然产生隐式评分
2. **零额外标注成本**：不需要 process reward model、不需要人工标注、不需要步骤级标签
3. **工程简单**：运行在标准 GRPO 训练器上，核心优化器无需改动；双 adapter 仅 2% 额外参数
4. **实验结果一致性强**：跨模型规模（0.6B→4B）、跨家族（Qwen3.5, Gemma 4）、跨领域（math→code），增益方向一致
5. **推理链缩短是涌现的**：不需要显式惩罚就能获得更短的推理链

### 局限与待验证

1. **单次训练运行**: 所有训练数据点均为单次运行，run-to-run 训练方差未被量化；报告的 delta 需要谨慎解读
2. **分歧是假设而非测量**: 两个 adapter 的"不同盲点"是训练设置的设计假设，互补性未通过级联胜率之外的指标量化
3. **Code 领域增益较小**: CodeContests 上的绝对值较小，且单元测试奖励更稀疏，推广到更嘈杂的领域尚待验证
4. **推理延迟**: 推理时是两次顺序生成的级联，增加了一倍的推理延迟
5. **Token 成本不完全对等**: 仅报告最终阶段长度，drafter 阶段长度未报告；challenger 的对手摘要预填充未被等量化
6. **文本带宽瓶颈**: 模型间仅通过文本交换信息，带宽极其有限；未来工作方向是 latent-space 通信
7. **竞争 vs 难度加权奖励塑形**: Conversion bonus 可以被解读为"难度加权奖励塑形"而非真正的博弈论竞争压力，实验未区分这两种解释
8. **Post-hoc 方向选择**: 级联的更好方向在 held-out 集上事后选择

### 待探索问题

1. 模型间能力差距多大会导致博弈退化？
2. 是否在更嘈杂的奖励领域（如对话、创意写作）仍然有效？
3. 两个 adapter 的分歧程度是否可以量化并主动优化？
4. 三个或更多模型的竞争是否比两模型博弈更有效？
5. 跨模型家族配对（如 Qwen + Gemma）的效果如何？

## 关联笔记

- [[GRPO]]: base algorithm, group-relative advantage
- [[Mixture-of-Agents]]: compared zero-training control, cooperative aggregation
- [[DeepMath]]: primary evaluation benchmark (hard split, difficulty-8)
- [[SPIN]]: related self-play method, plays current policy against past generations
- [[Self-Play]]: broader family of methods; Agon is cross-model rather than self-model
- [[R-Zero]]: self-evolving reasoning via self-play, related but single-model
- [[DAPO]]: dynamic sampling to handle degenerate GRPO groups
- [[Process Reward Model]]: the approach Agon avoids by using rival grading
- [[LoRA]]: adapter instantiation for dual-policy pair
- [[Overthinking in LLMs]]: the length pathology Agon targets
- [[Esperantix]]: same author's latent-space communication work (future direction)
- [[Prover-Verifier Games]]: related adversarial training paradigm

## 速查卡片

- **方法名**: Agon (希腊语 agon, "contest")
- **核心思想**: 两个模型通过 draft-and-challenge 互为隐式评分器，奖励超越对手
- **与 GRPO 的关系**: 在 GRPO 的 group-relative advantage 框架上，将对手信息注入上下文并在奖励中加入 conversion bonus
- **需要什么**:
  - 一个 verifier（可验证的 ground-truth 奖励）
  - 两个能力相当但行为不同的模型（本文通过双 LoRA adapter 实例化，2% 额外参数）
  - 可参考的问题集
- **不需要什么**: Process reward model, 步骤级标注, 人工标注, 额外的裁判模型
- **关键奖励公式**: `R = 2·correctness + correctness·(1 - opponent_correctness) + 0.5·format`
- **角色轮换**: 每个 optimizer step 后 drafter/challenger 角色互换，否则 pass@1 从 61 降到 52
- **主结果 (Qwen3-0.6B, DeepMath-hard)**:
  - Zero-shot: 23 | GRPO: 30 (+7) | Coop: 46 (+16) | **Agon: 61 (+31)**
  - Agon 达到约 **2x GRPO** 的 pass@1，是未训练 MoA 增益的约 **8x**
  - 最终阶段平均推理链长度: Agon 3.5k vs GRPO 8.1k
- **推理时**: 两阶段级联，两次顺序生成，选 post-hoc 较优方向
- **适用领域**: 数学推理 (DeepMath) + 竞赛编程 (CodeContests); 任何有干净可验证奖励的领域
- **未来方向**: Latent-space 通信替代文本交换
