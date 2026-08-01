---
type: concept
aliases: [过程奖励, process-based supervision, step-level reward, process-based feedback]
---

# Process Reward

## 定义
在 [[RLVR]] 中，对推理链的每个中间步骤进行评估和奖励的机制，提供步骤级密集监督信号。

## 数学形式
$$R_{\text{process}} = \frac{\text{correct steps}}{\text{total steps}}$$

每个步骤与参考答案比较（数值容差通常为 $10^{-5}$），匹配得 1，否则得 0。可附加长度惩罚防止过度冗长。

## 核心要点
1. 提供密集的训练信号，适合缺乏自我纠正能力的小模型
2. 鼓励显式的增量计算和忠实的逐步推理
3. 可能激励模型过度产生步骤或探索不必要的详细路径
4. 在数学领域有效，因为步级验证相对明确
5. 与 [[Outcome Reward|结果奖励]] 混合可缓解冗长问题

## 代表工作
- [[Reward Granularity in RLVR]]: 在 Qwen2.5-0.5B + GSM8K 上，纯过程奖励（63.73%）大幅优于纯结果奖励（53.75%）
- Lightman et al. (2023): 早期过程监督在数学推理中的探索
- Uesato et al. (2022): 比较过程与结果监督在数学文字题上的效果

## 相关概念
- [[Outcome Reward]]
- [[Reward Granularity]]
- [[RLVR]]
- [[GRPO]]
- [[Chain-of-Thought]]
