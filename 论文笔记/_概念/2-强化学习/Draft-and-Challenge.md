---
tags: [concept, rl, protocol, multi-agent]
created: 2026-07-09
---

# Draft-and-Challenge

## 定义

Draft-and-Challenge 是 Agon 框架的核心训练协议，一种**不对称、角色轮换的双模型交互范式**。每一步中，一个模型起草解答（draft），另一个模型阅读其摘要并尝试超越（challenge）。

## 协议流程

```
Step t (even):
  1. A drafts N solutions from plain prompt → standalone GRPO update
  2. B reads each A's solution summary → produces N challenge rollouts
  3. B receives competitive reward (correctness + conversion bonus)
  4. Both adapters updated

Step t+1 (odd):
  Roles swap: B drafts, A challenges
```

## 关键设计选择

1. **Summary 而非 raw trace**: Challenge 阶段看到的是解答摘要（post-reasoning 总结），而非原始推理链。原始 trace 被丢弃因为：
   - Token 量大、预填充成本高
   - 充满探索噪声
   - Summary 携带精炼的推导逻辑

2. **隐藏最终答案**: 作为防复制措施（尽管 worked summary 通常暗示了答案）

3. **角色轮换**: 每个 optimizer step 后 drafter/challenger 角色互换。不轮换时 pass@1 从 61 降至 52

4. **Per-rollout 不同对手**: 组内每个 challenger rollout 看到不同的对手 draft → 组内对手难度有差异 → conversion bonus 有梯度

## 与相关概念的关系

- 是 [[Agon]] 的训练协议实现
- 区别于 self-refinement：草稿来自**不同的竞争模型**，而非自身
- 区别于 multi-agent debate：**竞争性**而非合作性；是 RL 训练的一部分而非仅在推理时使用
- 区别于 prover-verifier：角色**对称轮换**，两个模型都做 draft 和 challenge

## 主论文

- [[Agon]] (Beliaev, 2026)
