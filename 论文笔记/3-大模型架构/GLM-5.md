---
title: "GLM-5: from Vibe Coding to Agentic Engineering"
method_name: "GLM-5"
authors: [GLM-5 Team (Zhipu AI & Tsinghua University, 184+ authors)]
year: 2026
venue: arXiv
tags: [llm, rl, agent, post-training, moe, dsa, sparse-attention, mla, mtp]
zotero_collection: 
image_source: online
arxiv_html: https://arxiv.org/html/2602.15763
paper_url: https://arxiv.org/abs/2602.15763
code_url: https://github.com/zai-org/GLM-5
model_url: https://huggingface.co/zai-org/GLM-5-FP8
created: 2026-07-09
---

# 论文笔记：GLM-5: from Vibe Coding to Agentic Engineering

## 元信息

- **论文标题**: GLM-5: from Vibe Coding to Agentic Engineering
- **作者团队**: GLM-5 Team (智谱AI + 清华大学，184+ co-authors)
- **发表时间**: 2026-02-17 (v1), 2026-02-24 (v2)
- **发表平台**: arXiv:2602.15763
- **代码仓库**: https://github.com/zai-org/GLM-5
- **模型权重**: https://huggingface.co/zai-org/GLM-5-FP8 (MIT License)
- **基础设施框架**: https://github.com/THUDM/slime
- **训练硬件**: 华为昇腾 (Ascend NPU) + 昇思 MindSpore
- **模型参数量**: 744B 总参数 / 40B 激活参数 (MoE)
- **模型架构**: Dense (前3层) + MoE (后77层), MLA + DSA 稀疏注意力
- **上下文长度**: 200K tokens (最大 202,752)
- **预训练数据量**: 28.5T tokens
- **协议**: MIT License (开源权重)

## 一句话总结

> 基于MoE+稀疏注意力+全异步RL训练栈的744B开源旗舰模型，从氛围编程迈向智能体工程。

## 核心贡献

1. **提出并验证 DSA (DeepSeek Sparse Attention)**：通过 Lightning Indexer + Top-k (k=2048) 选择机制，将长序列注意力从 O(L²) 降至 O(L·k)，KV Cache 减少 75%，推理速度提升 1.5-2x，长文本能力损失 < 0.5%，且仅需 20B tokens 中期训练即可适配 (对比 DeepSeek-V3.2 的 943B tokens)。

2. **构建全异步 Agentic RL 训练基础设施 (slime + SAO)**：将轨迹生成与模型训练完全解耦到独立 GPU 集群，引入 TITO (Token-in-Token-out) 消除重分词错位、C3PO++ 动态分区解决长尾阻塞、Direct Double-Sided Importance Sampling 替代 PPO clipping，实现异步训练下稳定收敛。

3. **提出 On-Policy Cross-Stage Distillation**：将前序训练阶段的最优 checkpoint 作为 Teacher，将 Teacher-Student 的 log-prob 比率直接转化为 PPO/GRPO 中的 advantage 信号，在多阶段 RL (Reasoning → Agentic → General) 过程中防止灾难性遗忘，且所有训练在统一 RL 引擎中完成。

4. **实现国产芯片全栈适配**：深度适配华为昇腾、摩尔线程、海光 DCU、寒武纪、昆仑芯、沐曦、燧原等 7 大国产芯片平台，实现 Day-0 部署支持，长序列场景部署成本降低约 50%。

5. **三项 MLA 性能突破**：(a) Muon Split -- 按注意力头拆分上投影矩阵并独立正交化，使 MLA 性能匹敌 GQA-8；(b) MLA-256 -- 头维度 192→256，头数减 1/3，解码计算量显著降低；(c) 参数共享 MTP -- 3 层共享参数的多 Token 预测，推测解码接受长度 2.76 (优于 DeepSeek-V3.2 的 2.55)。

6. **首次在 Artificial Analysis Intelligence Index v4.0 达到 50 分的开源模型**，在 SWE-bench Verified (77.8%)、HLE w/ Tools (50.4)、BrowseComp (75.9%)、AIME 2026 I (92.7%) 等核心基准上达到或接近闭源前沿水平。

## 问题背景

### 要解决的问题

1. **从 Vibe Coding 到 Agentic Engineering 的范式转变**：当前大模型在代码生成上表现出色 ("vibe coding")，但在端到端软件工程任务中 (理解需求、定位代码、修改、测试、调试的完整闭环) 仍远不及人类工程师。需要将模型能力从"单步代码补全"提升到"多步自主工程执行"。

2. **长上下文的高计算成本**：200K 上下文的密集注意力 (Dense Attention) 计算复杂度为 O(L²)，在 MoE 大模型中成为计算和显存瓶颈。滑动窗口等方法虽可降低开销，但存在无可避免的信息损失。

3. **同步 RL 训练的 GPU 利用率低下**：传统同步 PPO/GRPO 中，推理引擎和训练引擎共享同一 GPU 资源池，长 Agent 轨迹的 rollout 生成与梯度更新交替执行导致大量 GPU 空闲 (pipeline bubble)，利用率仅 20%-30%。

4. **多阶段 RL 的灾难性遗忘**：Reasoning RL → Agentic RL → General RL 的序列化训练中，后序阶段会冲刷前序阶段获得的能力。

### 现有方法的局限

- **Dense Attention**：O(L²) 计算量随上下文长度平方增长，不具备可扩展性。
- **Sliding Window Attention**：固定窗口外的 token 信息永久丢失，在需要跨长距离信息检索的任务上性能退化严重。
- **同步 RL (PPO/GRPO)**：轨迹生成与梯度更新串行执行，在长 Agent 任务中单个 rollout 可能耗时数分钟，导致训练效率极低。
- **Off-Policy 蒸馏**：使用静态数据集进行知识蒸馏，存在 exposure bias (训练-推理分布不匹配)，且无法有效防止 RL 训练中的遗忘。
- **标准重要性采样**：在异步 RL 设置下，需要保存历史策略快照用于 importance weight 计算，存储与计算开销巨大。

### 本文的动机

GLM-5 的目标是构建一个同时具备强大推理能力 (Reasoning)、自主智能体能力 (Agentic) 和代码工程能力 (Coding) 的统一模型，通过三项核心技术创新实现从"氛围编程"到"智能体工程"的范式跨越：DSA 解决长上下文效率瓶颈，异步 RL 架构解决训练效率瓶颈，跨阶段在线蒸馏解决多阶段遗忘问题。同时，通过全栈国产芯片适配，实现开源模型在自主可控算力上的大规模部署。

## 方法详解

### 模型架构总览

GLM-5 采用 **Dense + MoE 混合架构**，复用了 DeepSeek-V3/V3.2 的 MLA + MoE 基础设计，因此可原生运行于 vLLM、SGLang 等推理框架。

| 参数 | 规格 |
|------|------|
| 总参数量 | 744B |
| 激活参数量 | ~40B/token |
| 层数 | 80 (前3层 Dense, 后77层 MoE) |
| 专家数 | 256 routed experts + 1 shared expert |
| 激活专家数 | Top-8 (sigmoid gating) |
| 稀疏度 | ~5.9% (8/256) |
| 注意力机制 | MLA (Multi-Head Latent Attention) + DSA (DeepSeek Sparse Attention) |
| 上下文窗口 | 200K tokens (最大 202,752) |
| 最大输出长度 | 128K tokens |
| 预训练数据 | 28.5T tokens |
| 词表大小 | 沿用 GLM-4.5 词表 |

对比前代 GLM-4.5 (355B 总参 / 32B 激活 / 128 专家 / 128K 上下文)，参数规模扩大约 2x，激活参数增加 25%，专家数翻倍，同时减少层数以降低专家并行 (EP) 的通信开销。

### 核心模块

#### 1. DSA (DeepSeek Sparse Attention) — 稀疏注意力

**设计理念**：将传统密集的 O(L²) 注意力替换为"Lightning Indexer 快速粗筛 → Top-k 精准计算"的两阶段流水线。

**两阶段流程**：

- **Stage 1 — Lightning Indexer (轻量级索引器)**：对每个 query token，快速扫描所有历史 key token 并打分。Indexer 直接在 MLA 压缩后的低秩潜向量 (rank=512) 上操作，每个 head 独立评分后取均值，使用 FP8 精度计算，每 FLOP 成本远低于主注意力。

- **Stage 2 — Top-k Selector & Sparse Attention**：仅对 Indexer 打分最高的 k=2048 个历史 token 执行完整的 MLA 注意力计算，其余所有 token 全部跳过。对于 200K 上下文，k=2048 ≪ L，等价于将计算量降低约 100 倍。

**核心公式**：

Indexer 对 query token q 与候选位置 i 的兼容性评分:

$$
s_i = q \cdot W_i + b_i
$$

$$
g_i = \max(0, s_i) \quad \text{(ReLU gating, 代替 Softmax)}
$$

$$
\mathcal{I}_{\text{top-k}} = \underset{i \in \{1,\dots,L\}}{\text{Top-k}}(g_i, k=2048)
$$

稀疏注意力仅在筛选出的索引集合 $\mathcal{I}_{\text{top-k}}$ 上执行:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\mathcal{I}_{\text{top-k}}}^T}{\sqrt{d_k}}\right) V_{\mathcal{I}_{\text{top-k}}}
$$

**Indexer 参数配置**：

| 参数 | 值 |
|------|-----|
| `index_topk` | 2048 |
| `index_head_dim` | 128 |
| `index_n_heads` | 32 |
| `qk_nope_head_dim` | 192 |
| `qk_rope_head_dim` | 64 |
| `kv_lora_rank` | 512 |

**DSA 训练策略**：

采用两阶段继续预训练引入 DSA，无需从头训练:
1. **Dense Warm-up (密集预热)**：约 1000 步，保持密集注意力，学习率 5e-3，每步 14 条序列各约 202K tokens，建立全局语义表征。
2. **Smooth Transition & Sparse Training (平滑过渡与稀疏训练)**：逐步增大稀疏度，最终仅在 ~20B tokens 上完成适配 (DeepSeek-V3.2 需要 943B)。

**DSA 在 RL 中的关键洞察**：
- 必须使用确定性 `torch.topk` 算子，CUDA/TileLang 的非确定性 top-k 会在 RL 几步后导致熵急剧下降和性能崩溃。
- RL 期间冻结 Indexer 参数，加速训练并防止不稳定学习行为。

**效果**：
- KV Cache 开销降低 75%
- 推理速度提升 1.5-2x (long-sequence)
- 200K 上下文 GPU 成本降低 50%
- 长文本基准 (RULER@128K, RepoQA@128K) 性能损失 < 0.5%

#### 2. MLA (Multi-Head Latent Attention) 优化

**MLA 原生设计**：将 QA 投影到 2048 rank 的潜向量，KV 投影到 512 rank 潜向量 (+64 维 RoPE)，仅缓存压缩后的 KV 潜向量，生成时动态重建完整 K、V。相比 GQA，KV Cache 从 2048 维降至约 576 维。

**Muon Split**：标准 MLA 在 Muon 优化器下性能不如 GQA-8，因为全局正交化约束了各注意力头的功能分化。Muon Split 将上投影矩阵 ($W^{UQ}$, $W^{UK}$, $W^{UV}$) 按注意力头拆分为独立小矩阵，各自独立正交化：

$$
W^{UQ}_{\text{split}} = [W^{UQ}_1, W^{UQ}_2, \dots, W^{UQ}_H], \quad \text{每个 } W^{UQ}_h \text{ 独立正交化}
$$

效果：MLA 性能匹敌 GQA-8，注意力分数在预训练期间保持稳定无需裁剪。

**MLA-256 变体**：将单头维度从 192 增大到 256，注意力头数减少约 1/3。在训练量和参数量不变的前提下，解码阶段头数减少降低了总点积计算量。

#### 3. MTP (Multi-Token Prediction) 参数共享

训练时使用 3 个 MTP 层但共享参数 (而非每层独立)，保持草稿模型内存不变。推测解码接受长度达 2.76 (推测 4 步)，优于 DeepSeek-V3.2 的 2.55。

#### 4. Slime 异步强化学习基础设施

**核心架构**：将推理引擎 (持续生成轨迹) 与训练引擎 (异步消费轨迹更新) 完全解耦到独立 GPU 资源池，通过 Multi-Task Rollout Orchestrator 中央编排 1000+ 并发 rollout 任务。

**关键组件**：

- **C3PO++**：基于 Token Budget 的动态 Rollout 分区。将未完成的 rollout 缓存在推理池，由更新后的策略在下一轮继续生成，避免长尾样本阻塞。相比同步方法实现约 1.5x 端到端加速 (rollout 阶段 2.5x 加速)。

- **TITO (Token-in-Token-out)**：推理端直接传输 Token ID 而非文本，通过 TITO Gateway 确保采样动作与优化动作在"像素级"严格对齐，彻底规避异步传输中重分词导致的 token 边界错位和 action-reward 对齐错误。

- **AState**：零冗余 P2P 权重同步，万亿参数模型 10 秒内完成策略同步。

- **AMem**：GPU 内存管理库，支持 Memory Switching 和分布式多路径传输。

- **ASandbox**：无服务器沙箱引擎 (100ms 冷启动，5,000 QPS)，为 Agent RL 提供可验证的隔离执行环境。

#### 5. Agentic RL 算法 (基于 GRPO + IcePop + DIS)

**三段式序列化 RL 流程**：

1. **Reasoning RL**：基于 GRPO + IcePop 算法。显式区分训练策略 $\pi_{\text{train}}$ 和推理策略 $\pi_{\text{infer}}$，通过 IcePop 的 pop 机制过滤 mismatch ratio 过大的样本。

**IcePop 目标函数**：

$$
\nabla_{\theta} J_{\text{IcePop}}(\theta) \sim \mathbb{E}_{a \sim \pi_{\text{infer}}(\theta_{\text{old}})} \left[ \mathcal{M}\!\left(\frac{\pi_{\text{train}}(a; \theta_{\text{old}})}{\pi_{\text{infer}}(a; \theta_{\text{old}})}\right) \cdot \nabla_{\theta}\log \pi_{\text{train}}(a; \theta) \cdot \hat{A} \cdot r(a) \right]
$$

其中 $\mathcal{M}$ 为 pop 算子：当 mismatch ratio < $\alpha$ (0.5) 或 > $\beta$ (5) 时丢弃该样本。在 AIME25 上，IcePop 比标准 TIS 重要性采样提升 ~6%，比基线 GRPO 高出 14%+。

**Reasoning RL 混合领域配置**：
- 四域平衡训练: 数学、科学、代码、工具集成推理 (TIR)
- 奖励类型: 0/1 正确性 Reward
- GRPO 超参: KL 系数=0, 温度=1.0, group size=32, 学习率 2×10⁻⁶
- IcePop clipping: ε_low=0.2, ε_high=0.28

2. **Agentic RL**：全异步解耦框架。基于 Direct Double-Sided Importance Sampling (DIS) 替代标准 PPO clipping，使用 rollout 策略 $\pi_{\text{rollout}}$ 作为重要性权重分母 (而非维护历史策略快照)。

**Direct Double-Sided Importance Sampling**：

$$
r_t(\theta) = \frac{\pi_{\theta}(y_t \mid q, y_{<t})}{\pi_{\text{rollout}}(y_t \mid q, y_{<t})}
$$

Token 级双侧裁剪掩码:

$$
\text{mask}(r_t) = \begin{cases} 1 & \text{if } 1 - \varepsilon_{\ell} \leq r_t(\theta) \leq 1 + \varepsilon_{h} \\ 0 & \text{otherwise} \end{cases}
$$

不对称裁剪边界 (ε_ℓ 更紧, ε_h 更松)，鼓励探索同时防止策略坍缩。

**DP 感知路由 (DP-Aware Routing)**：通过一致性哈希最大化 KV 缓存复用，加速长上下文推理。

**任务环境**：10K+ SWE 环境 + Terminal + Search。

3. **General RL**：三维度混合奖励系统。

**混合奖励公式**:

$$
R_{\text{general}} = \alpha \cdot R_{\text{rule}} + \beta \cdot R_{\text{ORM}} + \gamma \cdot R_{\text{GRM}}
$$

| 维度 | 负责内容 | 信号特性 |
|------|----------|----------|
| Rule-based Reward | 硬约束 (格式、安全、事实锚定) | 零方差 / 确定性 |
| ORM (Outcome Reward Model) | 可自动验证的结果正确性 | 低方差 / 高效率 |
| GRM (Generative Reward Model) | 开放式质量与人类偏好 | 高方差 / 覆盖广 |

GRPO 超参: 学习率 3×10⁻⁶, group size=8, KL 系数=0, 最大长度 32,768 tokens.

#### 6. On-Policy Cross-Stage Distillation (跨阶段在线策略蒸馏)

**动机**：多阶段 RL 中，General RL 阶段会冲刷 Reasoning RL 和 Agentic RL 阶段获得的能力。

**核心思想**：不通过传统 KL 散度约束进行蒸馏，而是将 Teacher-Student 的 log-prob 比率直接转化为 advantage 信号，送入 PPO/GRPO 的目标函数：

$$
\hat{A}_{i,t}^{\text{distill}} = \text{sg}\!\left[ \log \frac{\pi_{\theta_{\text{teacher}}}^{\text{infer}}(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta}^{\text{train}}(y_{i,t} \mid x, y_{i,<t})} \right]
$$

**解读**:
- $\pi_{\theta_{\text{teacher}}}^{\text{infer}}$: 前序阶段最终 checkpoint (冻结) 的推理策略
- $\pi_{\theta}^{\text{train}}$: 当前学生模型的训练策略
- $\text{sg}[\cdot]$: Stop Gradient -- 表达式的值作为常数 advantage，梯度不传回 Teacher
- 若学生在某 token 上的概率接近 Teacher → $\hat{A} > 0$ → 正奖励 (强化)
- 若学生的概率远低于 Teacher → $\hat{A} < 0$ → 负惩罚 (抑制遗忘)

**On-Policy 采样**：学生模型自己生成轨迹 (on-policy)，Teacher 仅对这些真实轨迹评分。这直接解决了 off-policy 蒸馏中的 exposure bias 问题。

**参数设置**：蒸馏 batch size=1024，group size=1，Teacher 训练集 prompt 混合采样。

**统一训练引擎**：SFT、RL、蒸馏全部在同一 RL 基础设施 (slime) 中运行，无需独立蒸馏代码库。

### SFT 数据处理

**三种思考模式**：

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| Interleaved Thinking | 每次 action/tool call 前推理，形成 Thought→Action→Observation 循环 | Agent 任务 |
| Preserved Thinking | 跨轮保留 reasoning_content，不丢弃历史推理，`clear_thinking=false` | Coding/多轮 Agent |
| Turn-level Thinking | 同一 session 按轮次独立控制 thinking 开关 | 灵活成本控制 |

**SFT 数据格式**：完整 agent trajectory (thought → action → observation 循环)，loss 仅计算在模型生成的 token 上 (工具结果的 token 被 mask)。

**训练上下文**：SFT 最大上下文 202,752 tokens。

### 训练数据与流程总览

```
预训练 (27T tokens)
  │ Web Data: GLM-4.5 pipeline + DCLM Classifier + World Knowledge Classifier
  │ Code Data: 多平台快照，低资源语言分类器 (Scala, Swift, Lua)，28% 增量
  │ Math & Science: 严格无合成数据过滤，LLM 评分筛选
  ▼
中期训练 (~1.55T tokens)
  │ 上下文扩展: 32K (1T) → 128K (500B) → 200K (50B)
  │ DSA 适配: ~20B tokens
  │ SWE 序列: ~10M issue-PR pairs → ~160B unique tokens
  │ 长上下文增强: NextLong/EntropyLong 合成 + MRCR 多轮召回
  ▼
后训练 (渐进式 RL)
  │ SFT: 三类数据 (General Chat / Reasoning / Coding & Agent)
  │ Reasoning RL: GRPO + IcePop, 四域混合
  │ Agentic RL: 异步 SAO + TITO + DIS + DP-routing
  │ General RL: 混合奖励 (Rule + ORM + GRM)
  │ On-Policy Cross-Stage Distillation
  ▼
最终 GLM-5 模型
```

## 关键公式

### 1. DSA Indexer 评分公式

Indexer 对 query $q$ 在位置 $i$ 的兼容性评分:

$$
s_i = q \cdot W_i + b_i, \quad g_i = \max(0, s_i), \quad \mathcal{I}_{\text{top-k}} = \underset{i}{\text{Top-k}}(g_i, k=2048)
$$

- $q$: query token 的压缩表示
- $W_i, b_i$: Indexer 可学习参数 (FP8 精度, 32 heads, head_dim=128)
- $g_i$: ReLU 门控输出 (非 Softmax, 降低筛选前计算成本)
- $\mathcal{I}_{\text{top-k}}$: 得分最高的 k=2048 个历史位置的索引集合

### 2. 稀疏注意力计算

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K_{\mathcal{I}_{\text{top-k}}}^T}{\sqrt{d_k}} + M\right) V_{\mathcal{I}_{\text{top-k}}}
$$

- $K_{\mathcal{I}_{\text{top-k}}}, V_{\mathcal{I}_{\text{top-k}}}$: 仅从 top-k 索引集合中选取的 K、V 子矩阵
- $M$: 因果掩码 (causal mask)
- $d_k$: 单头维度 (192, MLA-256 中为 256)

### 3. IcePop 策略梯度

$$
\nabla_{\theta} J_{\text{IcePop}}(\theta) \sim \mathbb{E}_{a \sim \pi_{\text{infer}}(\theta_{\text{old}})} \left[ \mathcal{M}\!\left(\frac{\pi_{\text{train}}(a; \theta_{\text{old}})}{\pi_{\text{infer}}(a; \theta_{\text{old}})}\right) \cdot \nabla_{\theta}\log \pi_{\text{train}}(a; \theta) \cdot \hat{A} \cdot r(a) \right]
$$

- $\pi_{\text{train}}$: 训练策略 (含 KL 惩罚的 rollout 策略)
- $\pi_{\text{infer}}$: 推理策略 (纯采样策略, 温度=1.0)
- $\mathcal{M}(\rho)$: Pop 算子 -- 当 $\rho \notin [0.5, 5]$ 时丢弃样本
- $\hat{A}$: Group-wise 相对优势估计 (GRPO)
- $r(a)$: 任务奖励 (0/1 正确性)

### 4. Direct Double-Sided Importance Sampling (DIS)

$$
r_t(\theta) = \frac{\pi_{\theta}(y_t \mid q, y_{<t})}{\pi_{\text{rollout}}(y_t \mid q, y_{<t})}, \quad \text{mask}(r_t) = \mathbb{1}[1 - \varepsilon_{\ell} \leq r_t(\theta) \leq 1 + \varepsilon_{h}]
$$

- $\pi_{\text{rollout}}$: 实际生成轨迹的策略 (替代传统 $\pi_{\theta_{\text{old}}}$)
- $\varepsilon_{\ell}, \varepsilon_{h}$: 下界/上界裁剪阈值 (不对称, $\varepsilon_{\ell} < \varepsilon_{h}$)
- 超出裁剪范围的 token 被完整 mask (梯度归零)

### 5. On-Policy Cross-Stage Distillation Advantage

$$
\hat{A}_{i,t}^{\text{distill}} = \text{sg}\!\left[ \log \frac{\pi_{\theta_{\text{teacher}}}^{\text{infer}}(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta}^{\text{train}}(y_{i,t} \mid x, y_{i,<t})} \right]
$$

- $\text{sg}[\cdot]$: Stop Gradient 算子, Teacher 不参与梯度计算
- $\pi_{\theta_{\text{teacher}}}^{\text{infer}}$: 前序阶段 frozen checkpoint 的推理策略
- $\pi_{\theta}^{\text{train}}$: 当前 stage 学生模型的训练策略
- 正值 → 学生行为与 Teacher 一致 → 强化; 负值 → 偏离 → 惩罚

### 6. General RL 混合奖励

$$
R_{\text{general}} = \alpha \cdot R_{\text{rule}} + \beta \cdot R_{\text{ORM}} + \gamma \cdot R_{\text{GRM}}
$$

- $R_{\text{rule}}$: 基于确定性规则的硬约束 Reward
- $R_{\text{ORM}}$: Outcome Reward Model 输出 (低方差、高效率)
- $R_{\text{GRM}}$: Generative Reward Model 输出 (高方差、覆盖广)
- $\alpha, \beta, \gamma$: 权重系数 (未公开具体数值)

### 7. GRPO 目标函数 (通用形式)

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \min\!\Big( r_t(\theta) \hat{A}_t,\ \text{clip}\!\big(r_t(\theta), 1-\varepsilon_{\ell}, 1+\varepsilon_{h}\big) \hat{A}_t \Big) \right]
$$

- $r_t(\theta) = \pi_{\theta}(y_t \mid q, y_{<t}) / \pi_{\theta_{\text{old}}}(y_t \mid q, y_{<t})$
- $\hat{A}_t$: Group-wise relative advantage (GRPO 无 Critic)
- 双侧不对称裁剪: $[1-\varepsilon_{\ell}, 1+\varepsilon_{h}]$

### 8. Scaling Law (MoE)

从 GLM 系列的 scaling law 研究中得出的幂律关系:

$$
L \propto N^{-0.076} \cdot D^{-0.095} \cdot C^{-0.058}
$$

- $L$: 验证损失
- $N$: 模型参数量
- $D$: 训练数据量
- $C$: 计算预算

## 关键图表

### Figure 1: 整体训练流程

```
┌──────────┐    ┌──────────────┐    ┌─────────────────────────────┐
│  Base    │───▶│  Mid-Training │───▶│  Post-Training (渐进式对齐)  │
│ Training │    │  上下文扩展     │    │                             │
│ 27T tok  │    │ 4K→32K→128K  │    │ SFT → Reasoning RL           │
│          │    │ →200K        │    │  → Agentic RL → General RL   │
│          │    │ DSA适配(20B)  │    │  → On-Policy Cross-Stage     │
│          │    │ SWE序列(160B) │    │     Distillation             │
└──────────┘    └──────────────┘    └─────────────────────────────┘
```

### Figure 2: 三种思考模式 (Thinking Modes)

```
Interleaved:      Preserved:           Turn-level:
Thought₁          [Think Block₁]        Turn₁: 💭 ON
  ↓               Action₁               Turn₂: 💭 OFF
Action₁           [Think Block₂]        Turn₃: 💭 ON
  ↓               Action₂
Thought₂          (跨轮保留推理块)
  ↓
Action₂
```

### Figure 3: DSA 稀疏注意力架构

```
Dense Attention:                DSA Sparse Attention:
┌──────────────┐               ┌──────────────────────┐
│ Q · K^T       │               │ Lightning Indexer    │
│  (All tokens) │               │ 对所有 Token 快速打分  │
│  O(L²)        │               │ ReLU 门控 + FP8 运算  │
│               │               │          ↓           │
│               │               │ Top-k Selector       │
│               │               │ (k=2048)             │
│               │               │          ↓           │
│               │               │ Sparse MLA Attention │
│               │               │ 仅对 Top-k 计算       │
│               │               │ KV Cache -75%        │
└──────────────┘               └──────────────────────┘
```

### Figure 4: Slime 异步 RL 框架

```
┌──────────────────────────────────────────────────────────┐
│  Multi-Task Rollout Orchestrator                          │
│  (中央编排, 1000+ 并发 rollout)                             │
├──────────────────────┬───────────────────────────────────┤
│  Inference Engine    │         Training Engine            │
│  (持续生成轨迹)       │  ────▶  (异步消费轨迹, 梯度更新)    │
│                      │  ◀────  (K步后 P2P 权重同步)        │
│  C3PO++ 动态分区      │         AState 零冗余同步           │
│  TITO Gateway        │         DIS 双侧重要性采样          │
│  DP-Aware Routing    │         AMem 内存管理               │
│  ASandbox 沙箱       │                                   │
└──────────────────────┴───────────────────────────────────┘
```

### Table 1: 模型架构参数对比

| 参数 | GLM-4.5 | GLM-5 | 变化 |
|------|---------|-------|------|
| 总参数量 | 355B | 744B | +109% |
| 激活参数 | 32B | 40B | +25% |
| 层数 | — | 80 | — |
| 专家数 | 128 | 256 | +100% |
| 激活专家/Token | 8 | 8 | — |
| 训练 Token | ~14T | ~28.5T | +104% |
| 上下文长度 | 128K | 200K | +56% |
| 注意力 | MLA (GQA) | MLA + DSA | 新增 |

### Table 2: 核心 Benchmark 结果

| Benchmark | GLM-5 | GLM-4.7/4.6 | Claude Opus 4.5 | GPT-5.2 |
|-----------|-------|-------------|-----------------|---------|
| HLE (w/ Tools) | **50.4** | 42.8 | 43.4 | 45.5 |
| SWE-bench Verified | 77.8 | 73.8 | **80.9** | 80.0 |
| SWE-bench Multilingual | 73.3 | 66.7 | **77.5** | 72.0 |
| Terminal-Bench 2.0 | **56.2** | 41.0 | 59.3 | 54.0 |
| AIME 2026 I | 92.7 | 92.9 (4.7) | **93.3** | — |
| HMMT Nov. 2025 | 96.9 | — | — | 97.1 |
| GPQA-Diamond | 86.0 | 82.9 (4.6) | — | — |
| BrowseComp (w/ ctx mgmt) | **75.9** | 45.1 (4.6) | — | — |
| BrowseComp (w/o ctx mgmt) | 62.0 | 52.0 (4.7) | — | 37.0 |
| τ²-Bench | 89.7 | 75.9 (4.6) | — | — |
| Vending Bench 2 | $4,432 | $2,377 (4.7) | — | $3,591 |
| IF Bench | 72.0 | 43.0 (4.6) | — | — |
| ARC-AGI | 44.7 | — | — | — |
| Simple Bench | 53.2 | — | — | — |

### Table 3: MTP 推测解码接受率对比

| 模型 | 接受长度 (4步推测) |
|------|-------------------|
| GLM-5 (3层共享参数 MTP) | **2.76** |
| DeepSeek-V3.2 | 2.55 |

### Table 4: 预训练数据组成

| 数据类别 | 估计占比 | 核心方法 |
|----------|---------|----------|
| Code & Reasoning | 30-40% | 多平台快照, AST 去重, 低资源语言分类器 |
| Web (General + Knowledge) | 25-35% | DCLM Classifier, World Knowledge 蒸馏 |
| Math & Science | 10-15% | LLM 评分, 严格无合成数据过滤 |
| SWE Agentic (mid-training) | ~160B unique | Issue-PR pairs, commit diffs, repo 级拼接 |
| Long-context (natural + synthetic) | ~1T+ | NextLong/EntropyLong, MRCR 多轮召回 |

### Table 5: 国产芯片适配平台

| 芯片厂商 | 芯片类型 | 适配框架 | 适配状态 |
|----------|---------|----------|----------|
| 华为昇腾 (Ascend) | NPU | MindSpore, SGLang | Day-0 |
| 摩尔线程 (Moore Threads) | MTT S5000 GPU | SGLang | Day-0 |
| 海光 (Hygon) | DCU | DTK 自研软件栈 | Day-0 |
| 寒武纪 (Cambricon) | MLU | — | Day-0 |
| 昆仑芯 (Kunlunxin) | XPU | — | Day-0 |
| 沐曦 (MetaX) | GPU | — | Day-0 |
| 燧原 (Enflame) | GCU | — | Day-0 |

## 实验

### 数据集

**预训练数据 (27T tokens)**：
- Web Data: 基于 GLM-4.5 pipeline，增强 DCLM Classifier (句子嵌入质量评分) 和 World Knowledge Classifier (Wikipedia 对齐的细粒度知识抽取)
- Code Data: 多代码托管平台快照 + 含代码网页，去重 unique tokens 增加 28%，新增 160B unique tokens 的 GitHub Issue-PR 对，低资源编程语言 (Scala, Swift, Lua) 专用分类器
- Math & Science: 高质量网页+书籍+论文，改进的 PDF 解析和网页抽取 pipeline，LLM 评分筛选"高教育价值"内容，严格规避合成数据
- 词表: 沿用 GLM-4.5 词表

**中期训练数据 (~1.55T tokens)**：
- 上下文扩展: 32K (1T) → 128K (500B) → 200K (50B)
- DSA 适配: ~20B tokens (仅 1000 步预热)
- SWE 序列: ~10M issue-PR pairs → 160B unique tokens (repo 级拼接: issue + commit diff + source files)
- 长上下文增强: NextLong/EntropyLong 合成相似文本交错拼接 + MRCR 多轮召回数据

**后训练数据**：
- SFT: 三类训练数据 (General Chat, Reasoning, Coding & Agent), 三种 thinking mode 标注
- Reasoning RL: 四域 prompt (数学、科学、代码、TIR), 0/1 奖励
- Agentic RL: 10K+ SWE 环境, Terminal-Bench 环境, Web Search 环境
- General RL: 通用对话 prompt + 人类风格锚点 (专家撰写回复)

### 实现细节

**硬件与框架**：
- 训练集群: 华为昇腾 NPU
- 训练框架: 昇思 MindSpore
- RL 框架: slime (github.com/THUDM/slime)
- 推理框架: vLLM, SGLang, transformers
- 量化: 混合精度 W4A8 (部署), FP8 (训练中 Indexer 运算)

**预训练超参**：
- MoE Gating: Sigmoid gating, top-8 routing + 1 shared expert
- 前3层 Dense, 后77层 MoE
- 词表复用 GLM-4.5

**DSA 训练**：
- Dense warmup: 1000 步, lr=5e-3, 14 seq × ~202K tokens/step
- 稀疏过渡: 逐步提高稀疏度, 共 ~20B tokens
- 确定性 top-k: `torch.topk` (非 CUDA 非确定性实现)

**Reasoning RL**：
- 算法: GRPO + IcePop
- Group size=32, batch size=32, lr=2×10⁻⁶
- KL coeff=0, temperature=1.0
- IcePop: α=0.5, β=5, ε_low=0.2, ε_high=0.28
- 四域混合: 数学/科学/代码/TIR

**Agentic RL**：
- 算法: SAO (Single-Rollout Async Optimization)
- 异步架构: slime, DIS with 双侧裁剪
- TITO Gateway 确保 token 对齐
- DP-Aware Routing for KV cache reuse
- ~1000 步异步训练稳定收敛
- Keep recent k=5 policies for off-policy correction

**General RL**：
- 算法: GRPO
- Group size=8, lr=3×10⁻⁶, max len=32,768
- 混合奖励: Rule + ORM + GRM

**On-Policy Cross-Stage Distillation**：
- Batch size=1024, group size=1
- Teacher: 前序阶段 final checkpoint (SFT / Reasoning RL / Agentic RL)
- Student: 当前阶段训练策略
- 蒸馏 advantage 直接嵌入 GRPO 目标

**推理部署**：
- 上下文: 200K input + 128K output (max 202,752)
- vLLM/SGLang 原生支持 (复用 DeepSeek-V3 架构)
- 量化: W4A8 mixed precision
- 国产芯片: 7+ 平台 Day-0 支持

### 可视化结果

1. **DSA 消融实验**：DSA 相比 Dense Attention 在 RULER@128K、RepoQA@128K 等基准上性能损失 < 0.5%，同时推理时延降低 > 50%，GPU 成本减少 50%。

2. **IcePop 消融**：在 AIME25 上，IcePop > TIS (重要性采样) +6%, > 基线 GRPO +14%。

3. **异步 Agent RL 对比**：SAO (DIS + TITO) 在 SWE-bench Verified、BeyondAIME、IMOAnswerBench 上一致优于 GRPO 变体。BrowseComp 从 55.3% 提升至 62.0% (k=5 policies 窗口)。

4. **Cross-Stage Distillation 消融**：加入蒸馏后，在后续 RL 阶段中保留前序能力的衰减幅度显著减小。

5. **MTP 接受率曲线**：GLM-5 的 3 层共享参数 MTP 在 4 步推测下接受长度 2.76，优于 DeepSeek-V3.2 的 2.55。

6. **LMArena**：Text Arena 和 Code Arena 开源模型排名第一。

7. **Artificial Analysis Intelligence Index v4.0**：首个达到 50 分的开源权重模型。

8. **缩放定律曲线**：验证了从 GLM-4.5 (355B) → GLM-5 (744B) 的 MoE scaling，性能随着参数量、数据量、计算量三者协同增长。

## 批判性思考

### 优点

1. **系统工程的完整性**：GLM-5 不是单点技术突破，而是从预训练数据、模型架构 (DSA/MLA/MTP/MoE)、异步 RL 基础设施 (slime)、训练算法 (IcePop/DIS/GRPO)、蒸馏策略到推理部署 (国产芯片) 的全栈系统性工程，每个环节都有精心设计。

2. **DSA 的经济性**：仅需 20B tokens 中期训练 (vs DeepSeek-V3.2 的 943B) 即可完成 DSA 适配，说明通过合理 warm-up 策略可以大幅降低稀疏注意力的迁入成本。

3. **异步 RL 的实用主义设计**：TITO 直接传输 Token ID (而非文本) 规避重分词问题，DIS 使用 rollout 策略替代历史策略快照，C3PO++ 通过 token budget 动态分区 -- 这些设计直面工程痛点，务实而非追求理论优雅。

4. **On-Policy Distillation 的简洁性**：将教师信号转化为 advantage 注入 RL 目标，无需额外训练循环或独立蒸馏代码库，利用已有 RL 基础设施即可完成，思路优雅且工程成本低。

5. **全开源 + 国产化**：MIT License 开源权重 + 7 大国产芯片 Day-0 适配，在开源社区和自主可控算力两个维度都做出了重要贡献。

### 局限性

1. **DSA 在短序列上的收益有限**：DSA 的优势主要在长序列 (128K+) 上体现，在短序列场景下 Indexer 的额外开销可能使整体效率劣于 Dense Attention。

2. **异步 RL 的理论理解不足**：SAO 中的 DIS 方法和 k-policy 窗口机制更多是工程驱动的折中方案，缺乏严格的理论收敛保证。论文对异步策略滞后对最终策略质量的影响缺乏深入的理论分析。

3. **IcePop 的阈值依赖**：α=0.5、β=5 等阈值可能需要针对不同任务调整，缺乏自适应的阈值选择机制。

4. **GRM 的可靠性问题**：General RL 中使用 GRM (Generative Reward Model) 评分，虽然比 ORM 更难被 exploit，但其评分方差更大，且 GRM 自身的 bias 可能影响模型行为。

5. **MoE 路由坍缩风险**：256 专家的 MoE 训练中，专家路由塌缩 (部分专家被过度使用而其他专家闲置) 的风险论文未详细讨论。

6. **国产芯片性能天花板**：虽然适配了 7 大国产芯片，但国产芯片在单卡算力和集群互联带宽上相比 NVIDIA H100/B200 仍有差距，训练效率可能存在代际差距。

7. **评测的局限**：SWE-bench Verified 77.8% 仍显著低于 Claude Opus 4.5 的 80.9%，且部分核心基准 (如 BrowseComp) 的结果可能受上下文管理策略影响较大，不同设置间可比性存疑。

### 潜在改进方向

1. **IndexCache (GLM-5.2 已采纳)**：利用相邻层 Indexer 输出 70-100% 重叠的特性，仅在 Full 层计算 Indexer，Shared 层复用缓存，进一步移除 75% Indexer 计算开销 (prefill 1.82x, decode 1.48x 加速)。

2. **自适应 IcePop 阈值**：根据训练进度和 reward 方差动态调整 pop 阈值 α、β，减少人工调参。

3. **Multi-Agent RL**：将单 Agent 的异步 RL 扩展到多 Agent 协作场景，利用 Agent 间交互信号提供更丰富的训练反馈。

4. **DSA 的 MoE-aware Indexing**：利用 MoE 的专家路由信息辅助 Indexer 决策，进一步减少不相关 token 的计算。

5. **更细粒度的跨阶段蒸馏**：当前蒸馏在"阶段"粒度进行，可扩展为"能力维度"粒度 (分别保留数学推理、代码能力、Agent 规划等)，实现更精准的能力保留。

### 可复现性评估

- **权重开源** (MIT License): 可直接下载验证推理性能 ✓
- **代码开源** (GitHub): slime RL 框架开源, SGLang/vLLM 推理支持 ✓
- **训练数据**: 预训练数据组成描述详细但原始数据未公开 ✗
- **训练超参**: 大部分关键超参已公开 ✓
- **国产芯片依赖**: 基于华为昇腾训练，非国产芯片复现训练需要适配 ✗
- **训练规模**: 28.5T tokens 训练 + 744B 参数，仅少数机构具备复现条件 ✗
- **综合评估**: **推理可完全复现 (开源权重+推理框架)**，训练复现取决于算力和数据获取能力

## 关联笔记

### 基于
- [[DeepSeek-V3]]: 复用其 MLA + MoE 基础架构设计
- [[DeepSeek-V3.2]]: DSA 稀疏注意力的原始概念来源
- [[GRPO]]: 强化学习优化算法基础
- [[GLM-4.5]]: GLM 系列上一代模型，直接的前序工作
- [[slime]]: 清华大学 THUDM 开源的 RL 训练框架

### 对比
- [[Claude Opus 4.5]]: Agent/Coding 能力对标 (SWE-bench 80.9% vs GLM-5 77.8%)
- [[GPT-5.2]]: 综合能力对标 (HLE 50.4 vs 45.5)
- [[DeepSeek-V3.2]]: 架构同源对比 (MLA+MoE+DSA+MTP 技术栈)
- [[MiniMax-M2.5]]: SWE-bench 80.2% 接近 Claude Opus 4.5
- [[Kimi K2.5]]: BrowseComp 对标 (60.6% vs GLM-5 75.9%)

### 方法相关
- [[DSA (DeepSeek Sparse Attention)]]: 稀疏注意力核心方法
- [[MLA (Multi-Head Latent Attention)]]: 压缩 KV 缓存注意力
- [[MoE (Mixture of Experts)]]: 256 专家的路由架构
- [[GRPO (Group Relative Policy Optimization)]]: 去 Critic 的组相对策略优化
- [[On-Policy Distillation]]: 在线策略知识蒸馏
- [[SAO (Single-Rollout Async Optimization)]]: GLM-5.2 异步 RL 算法
- [[IndexCache]]: DSA 的跨层索引复用加速
- [[MTP (Multi-Token Prediction)]]: 多 Token 预测与推测解码
- [[TITO (Token-in-Token-out)]]: token 准确对齐传输

### 硬件/数据相关
- [[华为昇腾 (Ascend NPU)]]: 训练与推理芯片平台
- [[摩尔线程 MTT S5000]]: 国产 GPU 推理平台
- [[海光 DCU]]: 国产加速卡平台
- [[MindSpore]]: 昇思 AI 训练框架
- [[Software Heritage]]: 预训练代码数据来源

## 速查卡片

### 模型卡片
| 项目 | 内容 |
|------|------|
| 参数量 | 744B total / 40B active |
| 架构 | Dense(3L) + MoE(77L, 256专家, Top-8) |
| 注意力 | MLA + DSA (k=2048) |
| 上下文 | 200K in / 128K out |
| 训练数据 | 28.5T tokens |
| 训练硬件 | 华为昇腾 NPU |
| 许可证 | MIT |

### 训练流程
```
预训练 27T → 中期 1.55T (上下文+DSA+SWE) → SFT → 
Reasoning RL (GRPO+IcePop) → Agentic RL (SAO异步) → 
General RL (混合奖励) → On-Policy Cross-Stage Distillation
```

### 核心技术要点
- **DSA**: Lightning Indexer (ReLU 门控) + Top-k=2048 选择 → O(L²) → O(L·k)
- **IcePop**: 显式区分 π_train vs π_infer, pop 过滤 mismatch ratio ∉ [0.5, 5] 的样本
- **TITO**: 传输 Token ID 避重分词, 实现 lossless action-reward alignment
- **DIS**: 用 π_rollout 替代 π_old, 不对称双侧裁剪 [1-ε_ℓ, 1+ε_h]
- **On-Policy KD**: Teacher log-prob ratio → advantage signal in GRPO/PPO target
- **Muon Split**: 按头独立正交化 MLA 上投影矩阵 → MLA = GQA-8
