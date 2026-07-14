---
title: "Group Sequence Policy Optimization"
method_name: "GSPO"
authors: [Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin]
year: 2025
venue: arXiv
tags: [reinforcement-learning, grpo, sequence-level, importance-sampling, moe, qwen]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/2507.18071v2
created: 2026-07-14
updated: 2026-07-14
aliases: [GSPO, Group Sequence Policy Optimization]
---

# 论文笔记：Group Sequence Policy Optimization (GSPO)

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | Qwen Team, Alibaba Inc. |
| **作者** | Chujie Zheng*, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu*, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin |
| **发表** | arXiv:2507.18071, 2025年7月 (v2 修正) |
| **对比基线** | [[DeepSeekMath|GRPO]], [[PPO]] |
| **链接** | [arXiv](https://arxiv.org/abs/2507.18071) |

---

## 一句话总结

> GRPO 的 token 级重要性采样**从根本上就是错的**（每个分布只有 1 个样本），提出**序列级 importance ratio**，让同一序列所有 token 等权重，Qwen3 的基础 RL 算法。

---

## 核心洞察

### GRPO 的 Ill-posed 问题

GRPO 的核心问题是重要性采样 (IS) 的**误用**：

$$r_t(\theta) = \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})}$$

重要性采样要求：对**每个分布**有足够多的样本以校正分布偏移。但在 token 级：
- 每个 next-token 分布只在 $y_t$ 处有 **1 个样本**
- 无法做分布校正 → ratio 只是噪声而非有意义的权重

**后果**：
1. 噪声沿序列长度累积（O(T)，T 增长时恶化）
2. 裁剪机制**放大而非缓解**了问题
3. 长序列中产生**不可逆崩溃**—一旦发生，还原检查点和调参都无效

### GSPO 的核心思想

> **单位匹配原则**: 奖励是序列级的 → IS ratio 也应该是序列级的

$$\text{GRPO: }\underbrace{\frac{\pi_\theta(y_t)}{\pi_{\theta_{\text{old}}}(y_t)}}_{\text{每个 token 独立比}} \quad\Rightarrow\quad \text{GSPO: }\underbrace{\left(\frac{\pi_\theta(y)}{\pi_{\theta_{\text{old}}}(y)}\right)^{1/|y|}}_{\text{整个序列一个比}}$$

---

## 方法详解

### 序列级 Importance Ratio

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,y_{i,<t})}\right)$$

这实际上是所有 token 级 ratio 的**几何平均**。

### GSPO 目标函数

$$\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\left(s_i(\theta)\cdot A_i,\ \text{clip}(s_i(\theta), 1-\epsilon_l, 1+\epsilon_h)\cdot A_i\right)\right]$$

其中 **组优势**（同 GRPO）:

$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

### 梯度对比：GRPO vs GSPO

**GRPO 梯度**（Token 级，每个 token 权重不同）:

$$\nabla\mathcal{J}_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_i A_i\cdot\frac{1}{|y_i|}\sum_t \underbrace{\frac{\pi_\theta(y_t)}{\pi_{\theta_{\text{old}}}(y_t)}}_{\text{不等权重}}\cdot\nabla\log\pi_\theta(y_t)\right]$$

**GSPO 梯度**（序列级，所有 token 权重相同）:

$$\nabla\mathcal{J}_{\text{GSPO}} = \mathbb{E}\left[\frac{1}{G}\sum_i \underbrace{s_i(\theta)}_{\text{统一权重}}\cdot A_i\cdot\frac{1}{|y_i|}\sum_t \nabla\log\pi_\theta(y_t)\right]$$

关键差异：GSPO 的 $s_i(\theta)$ 从求和符号中提了出来—同一响应内的所有 token 获得**完全相同的梯度缩放因子**。

### GSPO-token：Token 级变体

对需要更细粒度优势调整的场景（如多轮 RL）：

$$s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_t|x,y_{<t})}{\text{sg}[\pi_\theta(y_t|x,y_{<t})]}$$

其中 $\text{sg}[\cdot]$ 是 stop-gradient。当 $A_{i,t} = A_i$ 对所有 t 成立时，GSPO-token 数值上等价于 GSPO。

### 裁剪阈值

因 ratio 定义不同，阈值数量级不同：

| 算法 | 左阈值 $\epsilon_l$ | 右阈值 $\epsilon_h$ |
|------|:------------------:|:------------------:|
| GRPO | 0.2 | 0.27 |
| **GSPO** | **3e-4** | **4e-4** |

> GSPO 裁剪阈值比 GRPO 小约 3 个数量级，因为序列级 ratio 的变化幅度远小于 token 级。

---

## GRPO vs GSPO 完整对比

| 维度 | GRPO (Token 级) | GSPO (序列级) |
|------|:--------------:|:-------------:|
| **IS 合法性** | ❌ 每个分布 1 个样本 | ✅ 对齐 IS 原理 |
| **重要性比** | $r_{i,t} = \frac{\pi_\theta(y_t)}{\pi_{\theta_{\text{old}}}(y_t)}$ | $s_i = (\frac{\pi_\theta(y)}{\pi_{\theta_{\text{old}}}(y)})^{1/|y|}$ |
| **梯度权重** | Token 间不等 → **高方差** | 所有 token **等权重** |
| **裁剪比例** | 0.13% token 被裁剪 | **15%** 被裁剪（但更高效） |
| **MoE 训练** | 需要 [[StabilizingRL\|Routing Replay]] | **不需要任何特殊处理** |
| **训练稳定性** | 可能 **不可逆崩溃** | **全程稳定** |
| **基础设施** | 需训练引擎重算 token 似然 | **可直接用推理引擎似然** |
| **精度不一致容忍度** | 低 (token 级对精度敏感) | **高** (序列级平均后噪声抵消) |

### 裁剪比例的反直觉现象

| 指标 | GRPO | GSPO |
|:---|:----:|:----:|
| 裁剪比例 | 0.13% | **15%** |
| 训练效率 | 低 | **高** |

> GSPO 裁剪了 **100 倍以上**的 token 却取得了更高的训练效率。这说明 GRPO 的 token 级梯度估计本质上有噪声—token 级 ratio 的变化主要由噪声驱动而非有意义的学习信号。

---

## 实验结果

### 设置

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-30B-A3B-Base (MoE, ~30B 总参, ~3B 激活) |
| 评估 | AIME'24, LiveCodeBench, CodeForces Elo |
| 训练 | 每批 4 个小批量 |
| GRPO 裁剪 | l=0.2, r=0.27 |
| GSPO 裁剪 | l=3e-4, r=4e-4 |
| GRPO 需要 Routing Replay | ✅ |
| GSPO 需要 Routing Replay | ❌ |

### 关键结果

| 指标 | GRPO | GSPO | 优势 |
|:----|:----:|:----:|:----:|
| **训练准确率** | 较低 | **更高** | 全程领先 |
| **AIME'24 Pass@1** | 基线 | **超越** | 持续提升 |
| **LiveCodeBench** | 基线 | **超越** | 差距扩大 |
| **CodeForces Elo** | 基线 | **超越** | 持续提升 |
| **训练曲线** | 有波动 | **平滑稳定** | 无崩溃 |

### MoE 稳定性对比

GRPO 的问题：MoE 模型 **~10% 激活专家**在梯度更新后改变，导致 token 级 ratio 剧烈波动。

| 方法 | MoE 无需特殊处理 | 专家波动容忍 |
|:----|:---------------:|:-----------:|
| GRPO | ❌ 需 Routing Replay | ❌ 低 |
| GRPO-R2 | —（有开销） | ✅ 可容忍 |
| **GSPO** | **✅ 不需要** | **✅ 高** |

GSPO 关注**序列似然**而非单个 token 似然—MoE 模型保持语言建模能力，序列似然不会剧烈波动。

---

## 批判性分析

### 优点
1. **理论正确**: 从根本上解决 IS 在 token 级的误用问题
2. **改动极小**: 只需修改 IS ratio 的计算方式，代码变化 ~5 行
3. **MoE 天然友好**: 无需 Routing Replay，降低工程复杂度
4. **基础设施简化**: 可直接使用推理引擎的似然值，无需重算

### 局限性
1. **序列级 ratio 敏感性**: $s_i$ 通过几何平均计算，单个极值 token 会影响整个序列
2. **GSPO-token 的妥协**: 多轮 RL 时需要 token 级变体，引入额外复杂性
3. **裁剪阈值极端**: 3e-4 和 4e-4 的阈值对其他框架可能不鲁棒
4. **评估基准局限**: 主要在 Qwen3-30B-A3B 上验证，泛化性待确认

---

## 关联笔记

### 前置
- [[DeepSeekMath|GRPO]] — GSPO 改进的起点
- [[PPO]] — 策略梯度框架

### 同系列（阿里 Qwen 团队）
- [[SAPO]] — 同期工作，软门控解决不同问题
- [[StabilizingRL]] — 形式化框架
- [[SAO]] — 清华，异步单 rollout

---

## 速查卡片

> [!summary] GSPO
> - **核心**: 序列级 importance ratio 替代 token 级
> - **方法**: $s_i = (\pi_\theta(y_i)/\pi_{\theta_{\text{old}}}(y_i))^{1/|y_i|}$，序列级裁剪
> - **优势**: 无需 Routing Replay，MoE 训练稳定，可直接用推理引擎似然
> - **结果**: Qwen3 基础 RL 算法，超越 GRPO
> - **影响**: 从 IS 原理上纠正了 GRPO 的 token 级误用

---

*笔记创建时间: 2026-07-14 | 深度版*
