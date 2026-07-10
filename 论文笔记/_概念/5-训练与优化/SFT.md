---
type: concept
aliases: [Supervised Fine-Tuning, 监督微调]
---

# SFT (Supervised Fine-Tuning)

## 定义
在预训练基座模型上，使用高质量的人工标注或合成生成的 (prompt, response) 对进行监督学习微调，使模型学会遵循指令格式和在特定领域生成高质量回复。

## 数学形式

标准因果语言模型损失 (仅计算 response token 上的 loss):

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_{\theta}(y_t \mid x, y_{<t})$$

其中 $x$ 为 prompt, $y$ 为 target response, prompt 部分的 token loss 被 mask。

## 核心要点
1. GLM-5 SFT 数据分三类: General Chat, Reasoning (含思维链), Coding & Agent
2. Agent 场景的 SFT 数据以完整 trajectory 为单位 (thought → action → observation 循环), 工具返回 token loss 被 mask
3. 三种 Thinking Mode: Interleaved (每步推理), Preserved (跨轮保留推理), Turn-level (按轮次开关)
4. 最大训练上下文: 202,752 tokens
5. SFT 阶段后直接进入 RL 训练，不单独评估 SFT 模型

## 代表工作
- [[GLM-5]]: 三类 SFT 数据 + 三种 thinking mode 联合训练
- [[InstructGPT]]: SFT + RLHF pipeline 的奠基工作
- [[LLaMA-Chat]]: LLaMA 系列的 SFT 对齐方案

## 相关概念
- [[RLHF (RL from Human Feedback)]]: SFT + RL 的经典 pipeline
- [[On-Policy Distillation]]: SFT 阶段的 Teacher 也参与跨阶段蒸馏
- [[Data Flywheel]]: SFT → RL → 数据合成 → SFT 的迭代优化闭环
