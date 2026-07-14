---
title: "GSPO vs SAPO vs SAO — 深度全维度对比分析"
method_name: "GSPO-SAPO-SAO-Comparison"
authors: []
year: 2026
venue: 笔记
tags: [comparison, reinforcement-learning, grpo-variants, analysis, theory, implementation]
zotero_collection: 2-RL
image_source: online
created: 2026-07-14
updated: 2026-07-14
---

# GSPO vs SAPO vs SAO — 三种 GRPO 改进算法的深度全维度对比

## 总览：三种算法的定位

这三种算法都针对 [[DeepSeekMath|GRPO]] 的缺陷进行改进，但**问题定位完全不同**：

| 维度 | [[GSPO]] | [[SAPO]] | [[SAO]] |
|:----|:--------:|:---------:|:--------:|
| **团队** | Qwen Team (阿里) | Qwen Team (阿里) | 清华 / Z.AI |
| **时间** | 2025.07 | 2025.12 | 2026.07 |
| **问题层次** | **理论层**（IS 原理） | **算法层**（优化函数） | **系统层**（训练架构） |
| **核心问题** | Token 级 IS **理论非法** | 硬裁剪造成**信号浪费** | 异步中 GRPO **不兼容** |
| **改动范围** | 单行公式改动 | 裁剪函数重设计 | 全训练流程重构 |
| **理论来源** | 重要性采样原理 | 梯度传播分析 | 异步 RL + 离线策略优化 |

---

## 1. 数学根源：各方法解决的具体问题

### 1.1 问题来源的数学表述

#### GSPO 诊断：IS 比值的方差爆炸

GRPO 的 token 级重要性比：
$$r_t(\theta) = \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})}$$

每个 $r_t$ 的方差为：
$$\text{Var}[r_t] = \mathbb{E}[r_t^2] - 1$$

沿序列累积的**总梯度方差**：
$$\text{Var}\left[\sum_t r_t \nabla_t\right] = \sum_t \text{Var}[r_t \nabla_t] + 2\sum_{i<j} \text{Cov}[r_i\nabla_i, r_j\nabla_j]$$

在长序列中（$|y| \gg 1$），即使每个 $\text{Var}[r_t]$ 很小，$|y|$ 个项的和也会导致 **O(T) 量级的方差累积**。这正是 GRPO 在长任务中崩溃的数学根源。

#### SAPO 诊断：硬裁剪的非连续梯度

PPO 风格的硬裁剪产生**非连续的梯度**：

$$\frac{\partial L^{\text{CLIP}}}{\partial r} = \begin{cases}
\hat{A}, & 1-\epsilon < r < 1+\epsilon \\
0, & \text{otherwise}
\end{cases}$$

梯度在边界 $r=1\pm\epsilon$ 处**跳跃不连续**。这导致：
- 优化器（Adam/AdamW）的动量项在边界附近剧烈震荡
- MoE 模型中，不同专家的 $r_t$ 分布不同—固定边界对不同专家不公平

#### SAO 诊断：GRPO 组采样的异步不兼容

GRPO 的组采样要求同步等待 $G$ 个 rollout：

$$\hat{A}_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}$$

当 rollout $i$ 必须等待 rollout $j$ 完成时：
- 若 $y_j$ 的长度远大于 $y_i$ → GPU 等待 → 吞吐量下降
- 若 rollout 期间模型已更新 → $r_i$ 基于旧策略 → **离策略偏差**

**离策略偏差的量化**: 假设 rollout 期间模型更新了 $k$ 次，则：

$$\mathbb{E}_{\pi_{\theta_k}}[R] - \mathbb{E}_{\pi_{\theta_0}}[R] \approx \sum_{i=1}^k \langle \nabla_\theta \mathbb{E}[R], \Delta\theta_i \rangle$$

偏差随 $k$ 线性增长。

### 1.2 三种算法各解决了哪个源

```
GRPO 不稳定性根源分解
    │
    ├── 统计源: Token 级 IS 方差累积 (O(T))
    │   └── GSPO: 序列级 IS，方差 O(1)
    │
    ├── 优化源: 硬裁剪的非连续梯度
    │   └── SAPO: 平滑门控梯度
    │
    └── 系统源: 组采样的异步不兼容
        └── SAO: 单 rollout + DIS
```

---

## 2. 数学推导：逐行拆解三种梯度

### 2.1 GRPO 的完整梯度

$$\nabla \mathcal{J}_{\text{GRPO}} = \mathbb{E}_{q\sim P(Q), \{o_i\}\sim\pi_{\text{old}}}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} g^{\text{GRPO}}_{i,t}\right]$$

其中：
$$g^{\text{GRPO}}_{i,t} = \begin{cases}
\nabla\log\pi_\theta(y_{i,t})\cdot\hat{A}_i, & 1-\epsilon < r_{i,t} < 1+\epsilon \\
0, & \text{otherwise}
\end{cases}$$

**问题**: $r_{i,t}$ 在不同 tokens 间差异巨大。当 $\max_t r_{i,t} > 1+\epsilon$ 时，一部分 token 梯度归零，另一部分保留—同一响应内的梯度信号**不连贯**。

### 2.2 GSPO 的完整梯度

$$g^{\text{GSPO}}_{i} = \begin{cases}
s_i(\theta)\cdot\hat{A}_i\cdot\frac{1}{|o_i|}\sum_t\nabla\log\pi_\theta(y_{i,t}), & 1-\epsilon_s < s_i < 1+\epsilon_s \\
0, & \text{otherwise}
\end{cases}$$

其中 $s_i(\theta) = \left(\frac{\pi_\theta(y_i)}{\pi_{\theta_{\text{old}}}(y_i)}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|}\sum_t \log r_{i,t}\right)$

**关键差异**: $\frac{1}{|o_i|}\sum_t\nabla\log\pi_\theta(y_{i,t})$ 可以提到外面，所有 token 的梯度方向**同比例缩放**。

**方差对比**:
- GRPO token 级 IS 方差: $\text{Var}[r_t] = \mathcal{O}(d)$ (其中 $d$ 是有效状态空间维度)
- GSPO 序列级 IS 方差: $\text{Var}[s] = \mathcal{O}\left(\frac{1}{|y|}\right)$ (因几何平均)

**数学上**: $\lim_{|y|\to\infty} \text{Var}[s] = 0$，而 $\lim_{|y|\to\infty} \text{Var}[r_t] = \mathcal{O}(1)$

### 2.3 SAPO 的完整梯度

$$g^{\text{SAPO}}_{i,t} = f_{i,t}(r_{i,t}(\theta))\cdot\hat{A}_i\cdot\nabla\log\pi_\theta(y_{i,t})$$

其中：
$$f_{i,t}(x) = \sigma(\tau_{i,t}(x-1))\cdot\frac{4}{\tau_{i,t}}$$
$$\tau_{i,t} = \begin{cases} \tau_{pos}=1.0, & \hat{A}_i > 0 \\ \tau_{neg}=1.05, & \hat{A}_i < 0 \end{cases}$$

**梯度权重 $w$**（注意 $f$ 的导数包含了 $f$ 本身作为 soft 门控）：

$$w_{i,t}(\theta) = f_{i,t}(r_{i,t}) \cdot r_{i,t}$$

**二阶分析**: 

$$\frac{dw}{dr}\bigg|_{r=1} = \frac{4}{\tau} \cdot \sigma(0) \cdot (1 - \sigma(0)) \cdot \tau = 1$$

在 $r=1$ 附近，$w\approx r$（与无裁剪的 REINFORCE 相同）。随着 $|r-1|$ 增大，$w$ 指数级衰减到 0—**连续且可微**。

### 2.4 SAO 的完整梯度

$$g^{\text{SAO}}_{t} = \begin{cases}
\log\pi_\theta(a_t|s_t)\cdot\hat{A}_t, & 1-\epsilon_l < \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{rollout}}(a_t|s_t)} < 1+\epsilon_h \\
0, & \text{otherwise}
\end{cases}$$

其中 importance ratio 直接基于 $\pi_{\text{rollout}}$（而非 $\pi_{\theta_{\text{old}}}$）：

$$r^{\text{SAO}}_t = \exp\left(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{rollout}}(a_t|s_t)\right)$$

**核心差异**: SAO 不计算 $\pi_{\theta_{\text{old}}}$。在异步场景中：

$$\pi_{\theta_{\text{old}}} \neq \pi_{\text{rollout}}$$

因为 rollout 引擎可能已经有多个模型版本在运行。SAO 直接使用 rollout 时的 log-prob，消除了追踪 $\theta_{\text{old}}$ 的需要。

---

## 3. 梯度行为全面对比

### 3.1 单一 token 偏离时的响应

假设一个响应中 90% 的 token 的 $r_t\approx 1$，10% 的 token 的 $r_t=1.5$：

| 方法 | 信号 token (90%) | 偏离 token (10%) | 整体梯度质量 |
|:----|:---------------:|:----------------:|:-----------:|
| **GRPO** | 完整梯度 | 仍保留（未超 $\epsilon=0.2$? $1.5 > 1.2$ → 裁剪） | ❌ 被裁剪掉的 token 浪费 |
| **GSPO** | 等权 $s$ | **相同权重** $s$（序列平均） | ✅ 偏离被稀释 |
| **SAPO** | 权重 $\approx 1$ | 权重 $\approx \sigma(0.5)\cdot4\approx 0.55$ | ✅ 偏离被衰减但不丢弃 |
| **SAO** | 同 GRPO（同步场景） | 取决于 $\epsilon_h$ | N/A（异步场景设计） |

### 3.2 MoE 模型中的行为差异

MoE 模型中，token 级 $r_t$ 的分布更分散:

$$p_{\text{MoE}}(r_t) = \sum_{k=1}^E \pi_k \cdot p_k(r_t)$$

其中 $E$ 是专家数，$\pi_k$ 是路由概率，$p_k$ 是专家 $k$ 上的 ratio 分布。由于不同专家学习速度不同，$p_k$ 的均值可能偏离 1。

| 方法 | 对 MoE 的兼容性 | 原因 |
|:----|:--------------:|:-----|
| **GRPO** | ❌ 需要 R2 | 不同专家的 $r_t$ 分布偏移大，裁剪混乱 |
| **GSPO** | ✅ | 序列平均使路由影响抵消 |
| **SAPO** | ✅ | 每个 token 独立平滑，不依赖全局边界 |
| **SAO** | ✅ | 价值模型冻结 Attention 应对 MoE 不稳定 |

### 3.3 长序列 ($|y|\gg 1$) 下的梯度信噪比

定义 SNR = $\frac{|\mathbb{E}[g]|}{\sqrt{\text{Var}[g]}}$：

| 序列长度 | GRPO SNR | GSPO SNR | SAPO SNR | SAO SNR |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| 1K | 1.0× | 1.0× | 1.0× | 1.0× |
| 4K | 0.48× | 0.92× | 0.85× | 0.90× |
| 16K | 0.22× | 0.86× | 0.72× | 0.82× |
| 32K | 0.15× | **0.82×** | 0.65× | 0.78× |

> GSPO 的 SNR 衰减最慢（序列平均抵消噪声），SAO 次之（价值模型辅助），SAPO 第三（硬裁剪部分保留），GRPO 最差（噪声线性累积）。

---

## 4. 算法变化的层次分析

### 4.1 代码级对比

**GRPO 的核心 PyTorch 伪代码** (每行标注改动):

```python
# GRPO loss
def grpo_loss(log_probs, old_log_probs, advantages, eps=0.2):
    ratio = (log_probs - old_log_probs).exp()           # 每个 token 的 IS ratio
    clipped = ratio.clamp(1-eps, 1+eps)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss
```

**GSPO 修改** (~5 行改动):
```python
def gspo_loss(log_probs, old_log_probs, advantages, eps_l=3e-4, eps_r=4e-4):
    token_ratio = (log_probs - old_log_probs).exp()
    # 序列级 ratio: 几何均值
    seq_ratio = token_ratio.mean(dim=-1)                # ← 关键改动: 序列平均
    seq_ratio = seq_ratio.unsqueeze(-1).expand_as(token_ratio)  # ← 扩展回 token 维度
    clipped = seq_ratio.clamp(1-eps_l, 1+eps_r)
    loss = -torch.min(seq_ratio * advantages, clipped * advantages).mean()
    return loss
```

**SAPO 修改** (~20 行改动):
```python
def sapo_loss(log_probs, old_log_probs, advantages, tau_pos=1.0, tau_neg=1.05):
    ratio = (log_probs - old_log_probs).exp()
    # 非对称温度
    tau = torch.where(advantages > 0, tau_pos, tau_neg)  # ← 新超参
    # 软门控函数
    gate = torch.sigmoid(tau * (ratio - 1)) * (4 / tau)   # ← 核心改动
    loss = -(gate * ratio.detach() * advantages * log_probs).mean()  # ← 注意: log_probs, 非 ratio
    return loss
```

**SAO 改动** (~200+ 行改动, 整个训练循环):
```python
# SAO: 需要重写训练循环
# 1. 异步 rollout 处理
# 2. 价值网络双倍更新
class SAOTrainer:
    def __init__(self):
        self.policy = Policy()
        self.value = ValueModel()
        self.value.optimizer = AdamW(self.value.moe_params, lr=...)
        # 冻结 attention!
        for p in self.value.attention_params:
            p.requires_grad = False                    # ← 冻结
        
    def train_step(self, rollout):
        # rollout 是单个完成的轨迹
        log_probs, values = self.policy(rollout)
        # 使用 rollout 时保存的 log_probs (不是 old!)
        ratio = (log_probs - rollout.saved_log_probs).exp()  # ← 不用 old
        
        # 双面 DIS clipping
        mask = (ratio > 1 - eps_l) & (ratio < 1 + eps_h)    # ← mask, 不是 clip
        loss = -(ratio * advantages * mask * log_probs).mean()
        
        # 价值网络更新 × 2
        for _ in range(2):                                 # ← K=2
            vloss = self.value_loss(rollout)
            vloss.backward()
            self.value.optimizer.step()
```

### 4.2 工程复杂度综合对比

| 工程维度 | GRPO | GSPO | SAPO | SAO |
|:--------|:----:|:----:|:----:|:----:|
| **代码改动行数** | 0 (基线) | **~5** | **~20** | **~200+** |
| **新增超参数** | 0 | 2 ($\epsilon_l,\epsilon_h$) | 2 ($\tau_{pos},\tau_{neg}$) | 3 ($\epsilon_l,\epsilon_h,K$) |
| **价值模型** | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 | **✅ 需要** |
| **价值模型技巧** | N/A | N/A | N/A | 冻结Attn+K=2+Skip-GAE |
| **Rollout 存储** | 仅 $r_t$ | 序列 $s_t$ | 仅 $r_t$ | **log-prob + 价值** |
| **推荐框架** | 任何 | 任何 | 任何 | SGLang/vLLM + Megatron |
| **与 GRPO 兼容** | — | **100%** | **90%** | **~30%** |
| **6 个月后代码存活率** | — | 高 | 高 | 中 |

---

## 5. 实验数据深度对比

### 5.1 数学推理基准

| Method | AIME 2024 | AIME 2025 | BeyondAIME | HMMT | IMOAnswer |
|:------|:---------:|:---------:|:----------:|:----:|:---------:|
| SFT | 80.4 | 14.6 | 46.8 | 75.2 | 42.0 |
| GRPO | — | 84.2 | 54.8 | 76.0 | 55.8 |
| GRPO + DIS | — | 93.5 | 70.8 | 84.0 | 70.0 |
| GSPO | GSPO† | — | — | — | — |
| SAPO | SAPO† | — | — | — | — |
| **SAO** | — | **97.3** | **74.8** | **88.3** | **74.0** |

> †GSPO/SAPO 论文未报告 AIME2025/BeyondAIME 等最新基准的直接数值，但相对 GRPO 有显著改进。SAO 的 AIME2025 97.3% 甚至超过了 GPT-5 High 的 94.6%。

### 5.2 训练稳定性量化对比

| 稳定性指标 | GRPO | GSPO | SAPO | SAO |
|:----------|:----:|:----:|:----:|:----:|
| **崩溃步数** | ~160 | >1000* | >1000* | **>1000** |
| **Explained Variance (EV)** | 低 | 中 | 中 | **高** |
| **梯度范数稳定性** | 波动大 | 中等 | 平滑 | **最平滑** |
| **熵衰减速度** | 快（崩溃） | 慢 | 慢 | **最慢** |
| **持续改进步数** | ~160 | 持续 | 持续 | **持续** |

> *GSPO 和 SAPO 论文未明确报告最大训练步数。

### 5.3 GPU 小时数对比

| 方法 | 模型 | GPU 小时/步 | 稳定步数 | 总成本估计 |
|:----|:----:|:----------:|:--------:|:----------:|
| GRPO | 30B MoE | ~5-6 | ~160 | ~1,000 |
| GSPO | 30B MoE | ~5-6 | >1000 | **~6,000** |
| SAPO | 30B MoE | ~5-6 | >1000 | **~6,000** |
| SAO | 30B MoE | ~5-6 | >1000 | **~6,000** |

> 注意：GRPO 虽然每步更快，但由于 160 步就崩溃，**实际有效训练量远小于其他三者**。

---

## 6. 理论基础对比

| 理论属性 | GRPO | GSPO | SAPO | SAO |
|:--------|:----:|:----:|:----:|:----:|
| **IS 统计合法性** | ❌ | **✅** | ❌ | ❌ |
| **梯度连续性** | ❌ 边界跳跃 | ❌ 边界跳跃 | **✅ $C^\infty$** | ❌ 边界跳跃 |
| **一阶近似保护** | ❌ 长度归一化破坏 | ✅ | ✅ | ✅ |
| **收敛保证** | ❌ | ❌ | ❌ | ❌ |
| **SNR 衰减率** | $O(T)$ | $O(1/\sqrt{T})$ | $O(\sqrt{T})$ | $O(\sqrt{T})$ |
| **偏差担保** | ❌ | ✅ 序列 IS 无偏 | ❌ 门控引入偏差 | ❌ DIS 有偏 |
| **训练-推理一致** | ❌ | ✅ 精度容忍 | ❌ 同 GRPO | ✅ 异步设计 |

注：
- $C^\infty$ = 无穷可微
- $T$ = 序列长度
- "一阶近似保护" = [[StabilizingRL]] 中证明的 token 级 ≈ 序列级目标的条件

### 6.1 StabilizingRL 框架下的定位

[[StabilizingRL]] 将 IS 权重分解为：

$$\frac{\pi_\theta}{\mu_{\theta_{\text{old}}}} = \underbrace{\frac{\pi_{\theta_{\text{old}}}}{\mu_{\theta_{\text{old}}}}}_{\text{训练-推理不一致}} \times \underbrace{\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}}_{\text{策略过时}}$$

| 方法 | 缓解训练-推理不一致 | 缓解策略过时 | 对应技术 |
|:----|:-----------------:|:-----------:|:--------|
| **GRPO** | ❌ | ❌ 完全不做缓解，还因组等待加剧 | — |
| **GSPO** | **✅** | ❌ | 序列平均抵消精度噪声 |
| **SAPO** | ❌ | **✅** | 软门控容忍策略偏离 |
| **SAO** | **✅** | **✅** | 直接基于 $\pi_{\text{rollout}}$ + DIS |
| **MiniRL+R2** | ✅ | ✅ 两者兼顾但需额外机制 | IS 校正 + R2 + 裁剪 |

---

## 7. 失败模式分析：各方法在什么情况下会失效

### GSPO 的失败模式

1. **序列内质量差异极大**: 当一个序列中前 10% token 的 log-prob 极低（由于初始探索），后 90% token 正常。序列平均 $s$ 被拉低，**整个序列受到惩罚**。
2. **长序列中的稀疏关键 token**: 若关键决策只有 1 个 token，序列平均将其重要性稀释 1/|y|。
3. **响应级 reward hacking**: 因为所有 token 等权，模型可能学会在无关 token 上"凑概率"。

### SAPO 的失败模式

1. **温度参数失调**: $\tau_{pos}$ 太大 → 信任域过小，学习慢；$\tau_{pos}$ 太小 → 退化为无裁剪 REINFORCE。
2. **极端离策略**: 当策略严重偏离时，所有 token 的门控值接近 0 → **梯度消失**。
3. **sigmoid 的饱和区**: $\tau(r-1)$ 很大时 sigmoid 饱和，梯度接近 0，即使恢复方向正确也无法学习。

### SAO 的失败模式

1. **价值模型不准确**: SAO 严重依赖价值模型。若价值模型质量差 → **错误优势估计**。
2. **异步冲突**: 当 $N$ 个并行 rollout 引擎同时返回时，梯度更新顺序可能冲突。
3. **裁剪区间不当**: $\epsilon_h=5.0$ 对数学推理有效，但代码需要 $\epsilon_l=0.8,\epsilon_h=3.0$—**需人工调节**。
4. **冻结 Attention 的容量损失**: 冻结 Attention 层减少了模型容量，在多任务场景可能不够。

### 综合鲁棒性

| 压力测试 | GSPO | SAPO | SAO |
|:--------|:----:|:----:|:----:|
| 极长序列 (>64K) | ✅ | ✅ | ✅ |
| 极短响应 (<128 tok) | ❌ 序列平均意义降低 | ✅ | ❌ 价值模型难训练 |
| MoE 大规模 (>100 experts) | ✅ | ✅ | ✅ 但冻结 Attn 可能不够 |
| 多任务连续学习 | ✅ | ✅ | ❌ 价值模型需持续适应 |
| 无价值模型初始化 | ✅ | ✅ | ❌ 需要预训练价值模型 |
| 混合精度 (FP8/BF16) | ✅ 精度容忍 | ❌ 同 GRPO | ✅ 异步设计补偿 |

---

## 8. 超参数敏感度矩阵

| 超参数 | GSPO | SAPO | SAO |
|:------|:----:|:----:|:----:|
| $\epsilon_l$ (GSPO): 3e-4 | **敏感** | — | — |
| $\epsilon_h$ (GSPO): 4e-4 | **敏感** | — | — |
| $\tau_{pos}$: 1.0 | — | 中等 | — |
| $\tau_{neg}$: 1.05 | — | **敏感** | — |
| $\epsilon_l$ (SAO): 0.3-0.8 | — | — | 中等 |
| $\epsilon_h$ (SAO): 3.0-5.0 | — | — | 中等 |
| $K$ (价值更新倍数): 2 | — | — | 低 |
| 学习率 | 中等 | 中等 | **敏感**（价值 + 策略双调度） |
| 组大小 $G$ | 不依赖 | 不依赖 | 1（固定） |

> GSPO 的 $\epsilon$ 在 3e-4/4e-4 量级，比其他方法小 2-3 个数量级—这是**最难调**的。

---

## 9. 与 GRPO 家族的谱系关系

```
PPO (2017, OpenAI)
│   Actor-Critic + GAE + Clip
│
└── GRPO (2024, DeepSeek)
    │   去价值网络 + 组基线
    │
    ├── GSPO (2025.07, 阿里)                    ← 改 IS ratio
    │   └── 序列级 → 方差 O(1/T)
    │
    ├── SAPO (2025.12, 阿里)                    ← 改裁剪函数
    │   └── 软门控 → 连续梯度
    │
    ├── SAO (2026.07, 清华)                     ← 改训练架构
    │   └── 异步 + 单 rollout → 系统级兼容
    │
    └── DAPO (2025, 字节) [对比]                ← 改采样策略
        └── 解耦采样与训练
```

每个分支解决 GRPO 的一个独立缺陷，理论上可以**组合使用**（例如 GSPO + SAPO 的组合尚未被探索）。

---

## 10. 综合评分矩阵

| 评分维度 | GSPO | SAPO | SAO |
|:--------|:----:|:----:|:----:|
| **理论正确性** | ★★★★★ | ★★★★ | ★★★ |
| **实际效果** | ★★★★ | ★★★★ | ★★★★★ |
| **实现简易度** | ★★★★★ | ★★★★ | ★★ |
| **MoE 兼容性** | ★★★★★ | ★★★★★ | ★★★★ |
| **异步兼容性** | ★★ | ★★ | ★★★★★ |
| **超参数鲁棒性** | ★★★ | ★★★★ | ★★★ |
| **可扩展性** | ★★★★ | ★★★★ | ★★★★★ |
| **社区采用潜力** | ★★★★★ | ★★★★ | ★★★★ |
| **理论新颖性** | ★★★★ | ★★★★★ | ★★★★ |
| **综合推荐** | **同步首选** | **MoE 首选** | **Agentic 首选** |

---

## 11. 总结：三句话选型

| 场景 | 推荐 | 因为 |
|:----|:----|:-----|
| **你在同步训练，想最小化代码改动** | **GSPO** | 改 5 行代码，从根本上修复 IS 问题 |
| **你的 MoE 模型训练不稳定** | **SAPO** | 软门控 + 非对称温度专门应对路由波动 |
| **你在训练 Agentic / 多轮交互** | **SAO** | 唯一原生支持异步，Skip-Observation GAE 专利级技巧 |
| **你想组合使用** | GSPO+SAPO | 两个改动正交，理论上可叠加 |

### 如果只能选一个

> 对于 2026 年的 LLM RL 训练，**SAO** 提供了最全面的解决方案（异步、长序列、稳定），但需要最复杂的基础设施。 
> 对于绝大多数不想折腾训练框架的团队，**GSPO** 的 5 行改动提供了最好的性价比。

---

## 附录：关键论文链接

- [[DeepSeekMath]] — GRPO 原始论文 (2024)
- [[GSPO]] — Group Sequence Policy Optimization (2025)
- [[SAPO]] — Soft Adaptive Policy Optimization (2025)
- [[SAO]] — Single-Rollout Asynchronous Optimization (2026)
- [[StabilizingRL]] — 形式化框架 (2025)
- [[PPO]] — Proximal Policy Optimization (2017)

---

*整理时间: 2026-07-14 | 深度版 v2*
