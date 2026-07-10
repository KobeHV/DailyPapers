---
title: "RL Post-Training Builds Compositional Reasoning Strategies"
method_name: "RL Post-Training"
authors: [Azwar Abdulsalam, Nishil Patel, Andrew Saxe]
year: 2026
venue: "arXiv / 2nd Workshop on Compositional Learning, ICML 2026 (Seoul)"
tags: [rl, reasoning, grpo, compositional-learning, mechanistic-interpretability, rewrite-grammar, rlvr, transformer]
zotero_collection: 
image_source: online
arxiv_html: https://arxiv.org/html/2607.07646
arxiv_abs: https://arxiv.org/abs/2607.07646
created: 2026-07-09
affiliation: "Gatsby Computational Neuroscience Unit, UCL"
domains: [cs.AI, cs.CL]
---

# 论文笔记：RL Post-Training Builds Compositional Reasoning Strategies

## 元信息

- **作者**: Azwar Abdulsalam, Nishil Patel, Andrew Saxe (Gatsby Computational Neuroscience Unit, University College London)
- **发表**: 2026-07-08 (arXiv preprint), accepted to the 2nd Workshop on Compositional Learning at ICML 2026, Seoul
- **领域**: cs.AI, cs.CL
- **URL**: https://arxiv.org/abs/2607.07646
- **方法名**: RL Post-Training (GRPO + binary reward)
- **关键词**: reinforcement learning, compositional reasoning, GRPO, rewrite grammar, mechanistic interpretability, RLVR

## 一句话总结

> RL post-training does not merely amplify latent primitive skills --- it composes them into genuinely new, reusable, higher-level reasoning strategies through a phased mechanism of procedural discovery and consolidation.

## 核心贡献

1. **RL reorganizes primitive competence through a phased compositional mechanism**: Phase 1 strengthens primitive reductions; Phase 2 discovers valid composed procedures (sequential and parallel compositions), which are then consolidated into a stable, reusable repertoire.

2. **Sequential compositions collapse ordered chains** of primitive contractions into single-step operations; **parallel compositions combine independent primitives** in a single rewrite step --- both emerging spontaneously from binary reward signals.

3. **RL vs RFT: the critical difference is selectivity, not exploration volume**: Rejection Fine-Tuning (RFT) produces many shortcut-like rewrites that are often invalid, whereas RL (GRPO) concentrates exploration into valid, reusable structure. RFT improves early but plateaus; RL continues improving.

4. **Pretraining ablations show compositional emergence is gated by procedural organization**: The base model must organize primitive competence into reduction procedures that RL can later compress. Mere exposure to primitives is insufficient.

5. **Fully observable controlled environment enables causal claims**: Using a rewrite-grammar domain where every trace can be audited, the paper provides mechanistic evidence that RL composes strategies, not just amplifies them.

## 问题背景

### 要解决的问题 (Does RL merely amplify latent skills or compose new strategies?)

The central question: when applying RL post-training (e.g., GRPO, PPO) to large language models, does the RL process merely **amplify/reweight** reasoning patterns already latent in the base pretrained model, or does it genuinely **compose** primitive skills into new, higher-level strategies that were not present before?

This question is critical for understanding the capabilities and limitations of RL post-training paradigms like those used in DeepSeek-R1, ChatGPT o1/o3, and Claude.

### 现有方法的局限

- **Rejection Fine-Tuning (RFT)**: Samples many completions, filters for correct ones, and fine-tunes on those. Early improvements but plateaus --- it cannot discover genuinely new strategies because it only selects from the existing distribution.
- **Process Reward Models (PRMs)**: Require dense, per-step supervision --- expensive to annotate and may not scale to truly novel reasoning patterns.
- **Black-box LLM analysis**: In real LLMs, the pretraining distribution is unknown, making it impossible to determine whether a post-training strategy is "new" or merely "amplified."
- **Lack of mechanistic understanding**: Prior work shows that RL improves benchmarks but cannot explain *how* it does so at the level of individual reasoning traces.

### 本文的动机

To resolve the amplification-vs-composition debate through a **controlled, fully observable environment** where:
1. The pretraining distribution is known and controlled.
2. Every generated rewrite can be audited at the trace level.
3. Causal claims can be made about what RL does to model behavior.
4. The comparison between RL and RFT can be isolated to the learning algorithm itself.

## 方法详解

### Rewrite Grammar Environment

The authors construct a synthetic **rewrite-grammar environment** --- a formal grammar where:
- **Expressions** are strings of symbols.
- **Rewrite rules** (primitives) define valid transformations from one expression to another, e.g., `A + B → C` or `(X Y) → Z`.
- **A reasoning trace** is a sequence of rewrite steps from an initial expression to a final simplified form.
- **Primitive rewrite chains** consist of individual rule applications performed one at a time.

The grammar is designed so that certain **composed rewrite rules** (applying multiple primitives at once) are valid shortcuts. These composed rules are *not* in the pretraining distribution --- they must be discovered.

### Pretraining on Primitive Rewrite Chains

A **Transformer model** is pretrained via next-token prediction on sequences of primitive rewrite chains:
- Input: `[initial expression] [SEP] [step_1 output] [SEP] [step_2 output] ... [SEP] [final answer]`
- The model learns to predict each subsequent expression given the current one, effectively learning the individual primitive rewrite operations.
- The pretraining data **only contains single-primitive rewrites** --- no composed/shortcut rewrites are present.
- This ensures that any composed strategy observed during post-training is genuinely novel.

### Trace-Based Reasoning Task

After pretraining, the model is post-trained on a **trace-based reasoning task**:
- Given an initial expression, the model must generate a full rewrite trace leading to a simplified final answer.
- The model generates autoregressively: it predicts the next rewrite step, then feeds it back for the subsequent step.
- **Reward**: Binary --- `r = 1` if the final answer is correct, `r = 0` otherwise. No per-step process reward or intermediate supervision.

Critically, **held-out problems** are constructed using compositions of primitives that require:
- More steps than any single training example (testing compositional length generalization).
- Combinations of primitives not seen together in pretraining (testing compositional novelty).

### RL (GRPO) and RFT Post-Training

**GRPO (Group Relative Policy Optimization)**:

For each input $x$, GRPO samples $G$ outputs $\{y_i\}_{i=1}^{G}$ from the current policy. The normalized reward for output $i$ is:

$$\tilde{r}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

The advantage $\hat{A}_{i,t} = \tilde{r}_i$ is shared across all tokens in output $i$. The GRPO objective maximizes:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}\hat{A}_{i,t},\ \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right)\hat{A}_{i,t}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

In the **binary reward setting** ($r \in \{0, 1\}$), this simplifies to **group preference optimization** --- increasing the likelihood of correct responses and penalizing incorrect ones proportionally to their relative standing within the group.

**RFT (Rejection Fine-Tuning)**: Samples $K$ completions per prompt, discards incorrect ones, and fine-tunes via standard next-token prediction on the correct traces. No policy gradient signal --- pure supervised learning on filtered outputs.

**Key comparison**: Both methods see the same number of generated samples and correct traces. The difference is in *how* they use the signal --- RL uses both correct and incorrect samples (contrastive signal), while RFT only uses correct ones (positive signal only).

### Trace Analysis Methodology

The paper introduces a **trace-level auditing framework**:
- For every generated reasoning trace, classify each rewrite step as:
  - **Primitive**: matching a single pretraining rewrite rule.
  - **Sequential composition**: collapsing multiple ordered primitive steps into one.
  - **Parallel composition**: combining multiple independent (non-interacting) primitives in one step.
  - **Invalid/shortcut**: a rewrite that does not correspond to any valid composition of primitives.
- Track the **frequency and accuracy** of each type across training steps.
- Identify **macro-rules** --- frequently reused composed procedures that become stable parts of the model's repertoire.
- Compute **consolidation metrics**: how often a discovered composed procedure is reused vs. being a one-off occurrence.

## 关键公式

### 1. GRPO Objective (Group Relative Policy Optimization)

$$\tilde{r}_i = \frac{r_i - \mu_G}{\sigma_G}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_G)^2}$$

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\left(\rho_{i,t} \hat{A}_{i,t},\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

其中 $\rho_{i,t} = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比率，$\hat{A}_{i,t} = \tilde{r}_i$ 是组归一化后的优势估计。

### 2. Binary Reward Formulation

$$r(y|x) = \begin{cases} 1 & \text{if } \text{final\_answer}(y) = \text{target}(x) \\ 0 & \text{otherwise} \end{cases}$$

### 3. Trace Composition Formalization

A rewrite step is a **sequential composition** if:
$$\exists r_1, r_2 \in \mathcal{R}_{\text{primitive}} : r_1 \circ r_2(s) = t \text{ and } r_1(s), r_2(r_1(s)) \text{ are both defined}$$

A rewrite step is a **parallel composition** if:
$$\exists r_a, r_b \in \mathcal{R}_{\text{primitive}} : r_a \perp r_b \text{ (independent subterms) and } r_{a \parallel b}(s) = t$$

其中:
- $s$ 和 $t$ 是前后表达式
- $\mathcal{R}_{\text{primitive}}$ 是原始单步重写规则集合
- $r_a \perp r_b$ 表示两个规则作用在不相交的子项上

### 4. RFT Objective (对比)

$$J_{\text{RFT}}(\theta) = \mathbb{E}_{(x, y^+) \sim \mathcal{D}_{\text{filtered}}}\left[-\log \pi_\theta(y^+ | x)\right]$$

其中 $\mathcal{D}_{\text{filtered}} = \{(x, y) | r(y|x) = 1\}$，即仅使用正确轨迹的监督学习。

## 关键图表

### Figure 1: Base Model pass@k on Held-Out Problems
- **X轴**: Sampling budget $k$ (number of completions sampled per problem)
- **Y轴**: pass@k (fraction of problems solved at least once in $k$ samples)
- **关键观察**: The pretrained base model has low pass@k even at large $k$, demonstrating that held-out compositional problems are genuinely hard --- solutions are not merely "low probability" but essentially absent from the pretrained distribution.

### Figure 2: RL vs RFT Performance Curves
- **X轴**: Training steps / number of samples
- **Y轴**: Pass rate / success rate on held-out problems
- **多条曲线**: RL (GRPO), RFT, Base model baseline
- **关键观察**:
  - RFT shows **early improvement** followed by a **clear plateau**
  - RL shows **slower initial progress** but **continued improvement** beyond the RFT plateau
  - The gap between RL and RFT widens over time

### Figure 3: Dynamics of Compositional Strategy Emergence
- **X轴**: Training steps
- **Y轴**: Frequency of each rewrite type (primitive, sequential composition, parallel composition, invalid shortcut)
- **分面**: Separate panels for RL and RFT
- **关键观察**:
  - **RL Phase 1** (early): Primitive reductions are strengthened and become more frequent/reliable
  - **RL Phase 2** (mid-late): Sequential compositions emerge, followed by parallel compositions
  - **RFT**: Primitive frequency increases but composed strategies never meaningfully emerge; shortcut rewrites proliferate instead

### Figure 4: Discovery and Consolidation of Macro Rules
- **可视化**: Heatmap or trajectory plot showing individual composed procedures
- **X轴**: Training steps
- **Y轴**: Individual macro-rule IDs
- **颜色/强度**: Frequency of usage
- **关键观察**: A small set of composed macro-rules are discovered, then repeatedly reused, forming a stable repertoire. Macro-rules that are discovered persist rather than being forgotten.

### Figure 5: Structured vs Unstructured Exploration
- **对比**: Distribution of rewrite *types* for RL vs RFT at equivalent sample budgets
- **关键观察**: RFT generates a long tail of diverse but mostly invalid shortcut rewrites. RL concentrates probability mass on a smaller set of valid, structured compositions.

### Figure 6: Pretraining Ablation Results
- **条件对比**: 
  - Full pretraining (primitive chains with procedural structure)
  - Ablation 1: Primitive exposure without chain structure (shuffled or isolated primitives)
  - Ablation 2: Different pretraining data composition
- **关键观察**: Compositional strategies emerge **only** when pretraining organizes primitives into reduction chains. Mere exposure to primitives (without procedural organization) fails to enable RL-driven composition.

### Table 1: Pass@k Comparison Across Methods
| Method | pass@1 | pass@10 | pass@100 | pass@1000 |
|--------|--------|---------|----------|-----------|
| Base Model (pretrained) | low | low | low | low-medium |
| RFT (post-trained) | medium | medium | medium | medium (plateau) |
| RL / GRPO (post-trained) | medium-high | high | very high | near-perfect |

### Table 2: Rewrite Type Distribution
| Rewrite Type | Base Model | RFT (final) | RL (final) |
|-------------|------------|-------------|------------|
| Primitive (%) | dominant | dominant | moderate |
| Sequential Composition (%) | near-zero | low | significant |
| Parallel Composition (%) | near-zero | near-zero | significant |
| Invalid Shortcut (%) | low | high | low |

## 实验

### Base model pass@k on held-out problems

**设定**: Evaluate the pretrained model's ability to solve held-out compositional problems by sampling $k$ completions and checking if any reach the correct answer.

**结果**: Even at large $k$ (e.g., $k=1000$), the pretrained model rarely solves held-out compositional problems. This establishes that solutions are not merely "buried" in the distribution --- they are effectively absent. The model has the primitive skills but cannot compose them spontaneously.

**意义**: Validates that the held-out problems genuinely require compositional reasoning beyond what the pretrained model can do, setting up a clean testbed for whether post-training can induce this ability.

### RL vs RFT comparison

**设定**: Post-train identical base models with either RL (GRPO) or RFT on the same trace-based reasoning task, controlling for compute budget (total samples generated).

**核心结果**:
- RFT achieves **faster initial improvement** but **plateaus** at a moderate success rate.
- RL achieves **slower initial improvement** but **continues to improve** beyond the RFT plateau, eventually far surpassing it.
- The performance gap is not explained by sample efficiency --- RL uses samples more effectively by learning from both correct and incorrect traces.

**Ablation**: Matching RL and RFT on the number of *correct samples seen* (not total samples) still shows RL superiority, confirming the advantage comes from the contrastive learning signal, not just exposure volume.

### Dynamics of compositional strategy emergence

**设定**: Track rewrite-type frequencies throughout RL post-training, categorizing each step as primitive, sequential composition, parallel composition, or invalid shortcut.

**Phase 1 --- Strengthening primitives (early training)**:
- The model first increases the reliability of individual primitive reductions.
- Primitive accuracy rises; composed strategies are absent.
- This phase establishes the "procedural ingredients" needed for later composition.

**Phase 2 --- Discovering compositions (mid-late training)**:
- Sequential compositions emerge first: the model learns to collapse `A→B→C` into `A→C` in one step.
- Parallel compositions follow: combining `f(A)→f'(A)` and `g(B)→g'(B)` into a single rewrite when A and B are independent subterms.
- Composed procedures rapidly increase in frequency once discovered.

**对比 RFT**:
- RFT increases primitive frequency but does not transition to Phase 2.
- RFT generates a large number of **invalid shortcut rewrites** --- the model tries to "guess" a shortcut without properly composing primitives.
- Without the contrastive signal from failed attempts, RFT cannot distinguish valid compositions from invalid shortcuts.

### Discovery and consolidation of macro rules

**设定**: Identify specific composed rewrite patterns (macro-rules), track their first appearance, and measure their subsequent reuse frequency.

**发现**:
- A **small set** of macro-rules accounts for the majority of compositional behavior.
- Once discovered, macro-rules are **not forgotten** --- they persist and are reused across different problems.
- The repertoire stabilizes: after an initial discovery phase, no new macro-rules appear, and existing ones are simply applied more reliably.
- This suggests RL converges to a **stable procedural repertoire** rather than continuously exploring new strategies.

**量化指标**:
- **Discovery rate**: Number of new macro-rules per training step (peaks early in Phase 2, then decays).
- **Consolidation ratio**: Fraction of compositional steps using previously-discovered macro-rules (increases monotonically, approaching 1.0).
- **Repertoire size**: Total number of unique macro-rules (saturates after Phase 2).

### Structured vs unstructured exploration

**设定**: Compare the entropy and validity of the rewrite-type distribution between RL and RFT.

**RL exploration pattern**:
- **Low entropy** in the space of rewrite *patterns* --- concentrates on a small set of structured compositions.
- **High validity**: composed rewrites are almost always correct application of primitives.
- The KL penalty to the reference model constrains exploration to the neighborhood of valid procedures.

**RFT exploration pattern**:
- **High entropy** in rewrite patterns --- samples from many diverse but invalid shortcuts.
- **Low validity**: many generated rewrites do not correspond to any valid primitive composition.
- Without a contrastive signal, the model cannot learn to avoid shortcuts.

**Implication**: The advantage of RL is **not** that it explores *more*, but that it explores *smarter* --- the policy gradient signal guides exploration toward structured, valid compositions and away from shortcuts.

### Pretraining ablation: what enables strategy emergence?

**设定**: Vary the pretraining data to test what properties are necessary for RL to induce compositional strategies.

**Ablation conditions**:
1. **Full pretraining**: Primitive rewrite chains presented as ordered reduction sequences (procedural organization).
2. **Shuffled primitives**: Same primitive rules but presented in random order without chain structure.
3. **Isolated primitives**: Individual primitive applications without context of multi-step reduction.
4. **Reduced diversity**: Fewer distinct primitive rules or shorter chains.

**关键结果**:
- RL induces compositional strategies **only** in the full pretraining condition.
- Shuffled and isolated conditions fail: the model has knowledge of primitives but cannot compose them during RL.
- The critical factor is not *whether* primitives are known, but whether they are organized into **reduction procedures** (chains) that RL can later compress and compose.
- This supports the "procedural ingredients" hypothesis: "The base model provides weak procedural ingredients; RL builds them into reliable higher-level strategies."

**对照实验**: Models pretrained with procedural organization show traces where Phase 1 visibly "reactivates" the chain structure before Phase 2 compresses it. Models without procedural organization never develop reliable primitives to build on.

## 批判性思考

### 优点

1. **Controlled environment enables causal claims**: By using a synthetic rewrite-grammar domain with known pretraining distribution, the paper can definitively answer the amplification-vs-composition question in a way impossible with black-box LLMs. The evidence that RL composes (not just amplifies) is causally grounded.

2. **Trace-level auditing provides mechanistic insight**: The categorization of each rewrite step (primitive, sequential, parallel, invalid) is a powerful analysis framework. It reveals the *how* of RL's improvement, not just the *what*.

3. **Clean RL vs RFT comparison**: By controlling for sample budget and correct-sample exposure, the paper isolates selectivity as the key differentiator. This has practical implications for choosing post-training algorithms.

4. **Phased dynamics are well-characterized**: The Phase 1 (strengthening) → Phase 2 (composition) progression is clearly documented and supported by multiple metrics.

5. **Pretraining ablation addresses a fundamental question**: Showing that procedural organization (not mere primitive exposure) gates compositional emergence connects post-training to pretraining in a principled way.

6. **Reproducible and well-scoped**: The synthetic environment allows exact replication of all results, and the claims are appropriately scoped to the domain.

### 局限性

1. **Rewrite-grammar is a toy domain**: The environment is deliberately simple. It is unclear whether the phased compositional mechanism generalizes to:
   - Natural language reasoning (math, code, logic).
   - Larger-scale Transformers (the paper likely uses small models).
   - Multi-modal or embodied reasoning tasks.

2. **Binary reward may oversimplify real RLVR**: In practice, verifiable rewards for math/code are binary, but for general reasoning tasks, rewards are often learned (reward models) or fuzzy (LLM-as-judge).

3. **Single task type**: Only one type of compositional reasoning (rewrite chains) is studied. Different compositional structures (e.g., tree-structured reasoning, recursive decomposition, backtracking) may show different dynamics.

4. **Small model scale**: The paper likely uses small Transformers (given the synthetic domain). Scaling laws for compositional emergence --- does the phased dynamic hold at billion-parameter scales? --- are unexplored.

5. **No comparison to process rewards**: While the paper argues binary outcome rewards suffice, it does not compare against process reward models (PRMs) to show whether dense supervision could accelerate Phase 1 or enable faster composition discovery.

6. **Pretraining ablation is coarse**: The ablation shows procedural organization is necessary, but does not characterize the *minimum* procedural structure needed. Would partial chain structure suffice? What about implicit procedural knowledge?

### 潜在改进方向

1. **Scaling to more complex compositional structures**: Extend the rewrite grammar to include nested compositions, conditional branching (if-then-else rewrites), and recursive self-composition, testing whether the phased dynamic generalizes.

2. **Multi-turn and iterative reasoning**: Study whether RL can compose strategies across multiple reasoning turns with environment feedback, moving beyond single-trace generation.

3. **Combining with process rewards**: Investigate whether a small amount of process supervision (e.g., on a subset of steps) could accelerate Phase 1 and reduce the sample cost of reaching Phase 2.

4. **Curriculum learning for composition**: Design principled curricula that introduce increasingly complex compositional problems during RL, potentially accelerating macro-rule discovery.

5. **Connection to mechanistic interpretability in real LLMs**: Develop methods to audit traces of real LLMs (e.g., via sparse autoencoders or probing) that can detect whether similar phased compositional dynamics occur.

6. **Theoretical model of the phase transition**: Derive a toy theoretical model that predicts the Phase 1→Phase 2 transition as a function of primitive reliability and reward signal strength.

### 可复现性评估

- **高可复现性**: The synthetic rewrite-grammar environment is fully specified and can be reimplemented. GRPO is a standard algorithm. Binary reward is trivial to compute.
- **潜在障碍**: Exact hyperparameters (learning rate, KL penalty coefficient $\beta$, group size $G$, model architecture details) may not be fully reported in the preprint. However, the qualitative results (phased dynamics, RL vs RFT gap) are likely robust to these choices.
- **建议**: Authors should release the rewrite-grammar generator code and training configurations.

## 关联笔记

- [[GRPO]]: Group Relative Policy Optimization --- the RL algorithm used for post-training in this paper. Key property: uses group-normalized advantages from binary rewards, providing contrastive signal without a learned reward model.
- [[RFT]]: Rejection Fine-Tuning --- the comparison baseline. Samples completions, filters for correct ones, and fine-tunes via supervised learning. Plagued by shortcut proliferation and early plateau.
- [[DeepSeek-R1]]: Real-world exemplar of RL post-training at scale, using GRPO-like algorithms with rule-based verifiable rewards for math and code reasoning.
- [[RLVR]]: Reinforcement Learning with Verifiable Rewards --- the broader paradigm encompassing GRPO, RFT, and related methods that use binary/scalar verifiable rewards rather than learned reward models.
- [[Compositional Reasoning]]: The target capability --- combining primitive skills into novel, higher-level reasoning strategies. Central to debates about LLM reasoning capabilities.
- [[Mechanistic Interpretability]]: The analysis methodology used in this paper --- trace-level auditing of individual model decisions to understand *how* behavior is generated.
- [[From f(x) and g(x) to f(g(x))]]: Contemporaneous work (Yuan et al., UIUC/Tsinghua) providing complementary evidence that RL composes existing skills into new ones, using a synthetic string transformation framework.
- [[How Does RL Post-Training Induce Skill Composition?]]: Related work (Park, Kaur, Arora, Princeton PLI) studying compositional generalization in countdown tasks with tree-structured reasoning.
- [[RL Perceptron]]: Earlier work (Patel et al., same lab) deriving closed-form ODEs for policy learning dynamics in high dimensions.
- [[ProcGen]]: Procedural generation --- related methodology for creating controlled environments with known ground-truth structure.

## 速查卡片

| 维度 | 内容 |
|------|------|
| **核心问题** | RL post-training: amplification vs composition? |
| **答案** | RL **composes** primitives into new higher-level strategies |
| **实验环境** | Fully observable rewrite-grammar domain |
| **模型** | Transformer pretrained on primitive rewrite chains |
| **任务** | Trace-based reasoning with binary final-answer reward |
| **RL算法** | GRPO (Group Relative Policy Optimization) |
| **对比基线** | Rejection Fine-Tuning (RFT) |
| **Phase 1** | Strengthen primitive reductions |
| **Phase 2** | Discover sequential + parallel compositions |
| **关键差异** | Selectivity (not exploration volume) separates RL from RFT |
| **RFT问题** | Produces invalid shortcuts; plateaus early |
| **Pretraining Gating** | Procedural organization of primitives is necessary for composition |
| **核心隐喻** | "Base model provides weak procedural ingredients; RL builds them into reliable higher-level strategies" |
| **发表** | arXiv 2026-07-08; 2nd Workshop on Compositional Learning, ICML 2026 |
| **实验室** | Gatsby Computational Neuroscience Unit, UCL |
| **同类工作** | Yuan et al. "From f(x) and g(x) to f(g(x))"; Park et al. "How Does RL Post-Training Induce Skill Composition?" |
