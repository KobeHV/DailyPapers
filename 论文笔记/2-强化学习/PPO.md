---
title: "Proximal Policy Optimization Algorithms"
method_name: "PPO"
authors: [John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov]
year: 2017
venue: arXiv
tags: [reinforcement-learning, policy-gradient, trust-region, clipping, rl-foundation]
zotero_collection: 2-RL
image_source: online
arxiv_html: https://arxiv.org/html/1707.06347v2
created: 2026-07-14
updated: 2026-07-14
aliases: [Proximal Policy Optimization, PPO-Clip, PPO-Penalty]
---

# 论文笔记：Proximal Policy Optimization Algorithms

## 元信息

| 项目 | 内容 |
|------|------|
| **机构** | OpenAI |
| **作者** | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov |
| **发表** | arXiv:1707.06347, 2017年8月 (v2) |
| **引用量** | > 25,000 |
| **对比基线** | [[TRPO]], A2C, ACER, CEM |
| **链接** | [arXiv](https://arxiv.org/abs/1707.06347) \| [OpenAI Blog](https://openai.com/blog/openai-baselines-ppo/) |

---

## 一句话总结

> 用**一阶裁剪目标函数**替代 TRPO 的二阶 KL 约束，在保持稳定性的同时大幅简化实现，成为深度 RL 的事实标准算法。

---

## 问题背景

### 要解决的问题
策略梯度方法如何在**不破坏已有性能**的前提下进行尽可能大的策略更新？

### 现有方法的局限

- **Vanilla Policy Gradient (REINFORCE)**: 每个样本只做一次梯度更新，样本效率低。对步长极其敏感：步长太小收敛慢，步长太大策略崩溃。
- **Natural Policy Gradient / [[TRPO]]**: 使用二阶信息（Fisher 信息矩阵）构造 KL 散度约束，保证每次更新在"信赖域"内。但**实现极其复杂**（共轭梯度法、二次近似），计算昂贵，不兼容 dropout 和参数共享。

### 本文的动机
找到一个**既简单又稳定**的方法 — 兼具 TRPO 的可靠性约束和 SGD 的简便性。

---

## 方法详解

### 符号定义

| 符号 | 含义 |
|------|------|
| $\pi_\theta$ | 参数为 $\theta$ 的随机策略 |
| $a_t, s_t$ | $t$ 时刻的动作和状态 |
| $r_t(\theta) = \frac{\pi_\theta(a_t\|s_t)}{\pi_{\theta_{\text{old}}}(a_t\|s_t)}$ | **概率比**（importance ratio），$\theta_{\text{old}}$ 时为 1 |
| $\hat{A}_t$ | $t$ 时刻的优势函数估计 |
| $\epsilon$ | 裁剪阈值（默认 0.2） |

### PPO-Clip：裁剪替代目标

**核心公式**:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**动机与行为分析**:

- 第一项 $r_t(\theta)\hat{A}_t$ 是保守策略迭代 (CPI) 的目标。
- 第二项将 $r_t$ 裁剪到 $[1-\epsilon, 1+\epsilon]$ 内，消除将 $r_t$ 移出区间带来的梯度。
- 取 $\min$ 使最终目标是 CPI 的**下界**（悲观边界）— 当更新有风险时忽略，有收益时保留

**四种情况的具体行为**:

| 情况 | 条件 | 梯度行为 |
|------|------|----------|
| **正优势，未裁剪** | $\hat{A}>0,\ r_t \leq 1+\epsilon$ | 正常提高 $r_t$（增加该动作概率） |
| **正优势，已裁剪** | $\hat{A}>0,\ r_t > 1+\epsilon$ | **梯度为 0**，防止过度提高 |
| **负优势，未裁剪** | $\hat{A}<0,\ r_t \geq 1-\epsilon$ | 正常降低 $r_t$（减少该动作概率） |
| **负优势，已裁剪** | $\hat{A}<0,\ r_t < 1-\epsilon$ | **梯度为 0**，防止过度降低 |

**直观解释**: 算法好比一个"橡皮筋" — 在安全范围内自由移动，但一旦拉伸超过边界就停止用力。

### PPO-Penalty：自适应 KL 惩罚

作为裁剪的替代方案：

$$L^{KL-PEN}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat{A}_t - \beta\cdot\text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]\right]$$

**自适应调整 $\beta$**:
- 计算 $d = \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_{\text{old}}}, \pi_\theta]]$
- 如果 $d < d_{\text{targ}}/1.5$: $\beta \leftarrow \beta/2$（KL 太小 → 放松惩罚）
- 如果 $d > d_{\text{targ}}\times 1.5$: $\beta \leftarrow \beta \times 2$（KL 太大 → 加强惩罚）
- 目标 KL 典型值：$d_{\text{targ}} = 0.01$

> **经验结论**: 裁剪版本比自适应 KL 版本更简单、表现更好。论文主要推荐 PPO-Clip。

### 完整多目标函数

当策略和价值网络共享参数时：

$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\right]$$

其中：
- $L_t^{VF} = (V_\theta(s_t) - V_t^{\text{targ}})^2$：价值网络的均方误差损失
- $S[\pi_\theta](s_t)$：策略熵奖励项（鼓励探索）
- $c_1, c_2$：权重系数（典型值 0.5, 0.01）

### 优势估计：GAE

对长度为 $T$ 的轨迹段使用截断 GAE：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

| 参数 | 含义 | 典型值 |
|------|------|--------|
| $\gamma = 0.99$ | 折扣因子 | 控制长期 vs 短期 |
| $\lambda = 0.95$ | GAE 平滑参数 | 平衡偏差与方差 |

### 完整算法伪代码

```
1: for iteration = 1, 2, ... do
2:     for actor = 1, ..., N do                           // N 个并行 actor
3:         在环境中运行策略 π_θ_old 执行 T 个时间步
4:         计算优势估计 Â_1, ..., Â_T（使用 GAE）
5:     end for
6:     计算裁剪替代目标 L_CLIP(θ)
7:     优化 L_CLIP 共 K 个 epoch，小批量大小 M ≤ NT   // 关键创新！
8:     θ_old ← θ                                        // 更新行为策略
9: end for
```

> **核心创新在第 7 行**：标准策略梯度方法每个数据样本只做 1 次更新，PPO 通过裁剪机制安全地支持 K 个 epoch 的小批量更新（典型值 K=3~15），大幅提升样本效率。

---

## 关键结果

### 实验设置

| 环境 | MuJoCo (连续控制) | Atari (离散控制) |
|------|:----------------:|:----------------:|
| 网络 | 2层 MLP 64 单元, tanh | CNN (Nature 架构) |
| 视野 T | 2048 | 128 |
| 学习率 | 3e-4 | 2.5e-4 (退火) |
| epoch K | 10 | 3 |
| 小批量 | 64 | 32 × 8 |

### 替代目标对比 (MuJoCo, 1M 步)

| 算法 | 平均标准化得分 |
|------|:--------------:|
| 无裁剪/无惩罚 | -0.39 ❌ |
| **PPO-Clip, ε=0.2** | **0.82** ✅ |
| PPO-Clip, ε=0.1 | 0.76 |
| PPO-Clip, ε=0.3 | 0.70 |
| 自适应 KL, d=0.003 | 0.68 |
| 自适应 KL, d=0.01 | 0.74 |
| 自适应 KL, d=0.03 | 0.71 |

> **ε=0.2 是最优选择**，对超参数不敏感。

### 与基线全面对比 (Atari 49 游戏)

| 指标 | A2C | ACER | **PPO** |
|:----|:---:|:----:|:-------:|
| 全训练期平均每轮奖励 | 1 | 18 | **30** |
| 最后 100 轮平均每轮奖励 | 1 | 28 | **19** |

PPO 在以下游戏中取得了 A2C **零分**而 PPO 大幅进步的成果：
- Enduro: 0 → **758.3**
- Freeway: 0 → **32.5**
- Zaxxon: 16.3 → **5008.7**

### 3D 人形机器人 (Roboschool)

PPO 在三个高维任务上均表现出稳定的学习能力，包括 FlagrunHarder 任务（被方块击倒后需要从地面爬起）。

---

## 批判性分析

### 优点
1. **实现极简**: 核心修改只需对标准策略梯度代码添加 5 行裁剪逻辑
2. **鲁棒性强**: ε=0.2 在几乎所有任务上都表现良好，无需精细调参
3. **样本效率高**: 多 epoch 小批量更新大幅提升数据利用率
4. **通用性广**: 可用于连续/离散控制，兼容 RNN、共享参数等架构

### 局限性
1. **裁剪信息丢失**: 当 $r_t$ 超出区间时梯度完全归零，浪费了部分学习信号（后续 [[SAPO]] 用软门控解决此问题）
2. **仍需价值网络**: 价值网络和策略模型一样大，增加内存（[[DeepSeekMath|GRPO]] 去掉了价值网络）
3. **对 LLM 场景不直接适用**: 
   - 需要每个 token 的奖励信号（LLM 通常只在序列末尾有奖励）
   - 训练-推理 precision 差异导致 IS 权重有偏（[[StabilizingRL]] 形式化分析此问题）
4. **On-policy 样本效率低**: 每次更新后丢弃旧数据

### 后续工作演化

```
PPO (2017) ─── Actor-Critic + Clip + GAE
  └── GRPO (2024) ─── 去价值网络 + 组基线
      ├── GSPO (2025) ─── 序列级 IS ratio
      ├── SAPO (2025) ─── 软门控替代硬裁剪
      ├── SAO (2026) ─── 异步 + 双面 DIS
      └── StabilizingRL (2025) ─── 形式化框架
```

---

## 关联笔记

### 前置基础
- [[TRPO]] — PPO 的理论基础，信赖域策略优化
- [[GAE|Generalized Advantage Estimation]] — PPO 使用的优势估计器
- Policy Gradient Theorem — 策略梯度的理论依据

### 后续改进
- [[DeepSeekMath|GRPO]] — 去掉价值网络，用组基线替代
- [[StabilizingRL]] — 形式化 PPO/GRPO 在 LLM 中的训练稳定性
- [[SAO]] — 异步场景下的 PPO 改进

### 对比方法
- A2C — 同步版本的 A3C，PPO 的对比基线
- ACER — 高效 Actor-Critic，PPO 的对比基线

---

## 速查卡片

> [!summary] PPO
> - **核心**: 裁剪替代目标实现稳定策略更新
> - **公式**: $L^{CLIP} = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t)]$
> - **超参数**: $\epsilon=0.2$, $\gamma=0.99$, $\lambda=0.95$, K=3~10
> - **结果**: MuJoCo 7/7 环境 SOTA, Atari 49 游戏 30 胜
> - **影响**: RL 领域引用量最高 (>25000) 的基础算法之一

---

*笔记创建时间: 2026-07-14 | 深度版*
