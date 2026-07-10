---
type: concept
aliases: [LoRA, Low-Rank Adaptation, 低秩适配]
---

# LoRA

## 定义
LoRA（Low-Rank Adaptation）是一种参数高效的微调方法（PEFT），通过向预训练权重注入可训练的低秩分解矩阵来适配下游任务，冻结原始权重。

## 数学形式
$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$$
前向传播时：
$$h = W'x = Wx + BAx$$

## 核心要点
1. 冻结原始权重，只训练低秩矩阵 A 和 B
2. 推理时可合并到原始权重中，无额外延迟
3. 秩 r 通常取 8-64，参数量仅为主模型的 0.1%-1%
4. 常与量化结合使用（QLoRA）
5. 适用于注意力层和 FFN 层

## 代表工作
- [[LoRA: Low-Rank Adaptation of Large Language Models]]: 原始论文
- [[QLoRA]]: LoRA + 4-bit 量化
- [[DoRA]]: LoRA 的权重分解变体
- [[PiSSA]]: 用 SVD 初始化 LoRA 矩阵
- [[LlamaFactory]]: 支持 LoRA 等多种 PEFT 的统一框架

## 相关概念
- [[SFT]]: 全参数微调 vs LoRA 参数高效微调
- [[量化]]: 常与 LoRA 联合使用
- [[MoE]]: 另一种参数高效利用方式
