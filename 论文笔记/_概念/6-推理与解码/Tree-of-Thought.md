---
type: concept
aliases: [ToT, Tree of Thoughts, 思维树]
---

# Tree-of-Thought (ToT)

## 定义
一种增强 LLM 推理的框架，在推理过程中生成多个思路分支并系统地探索，而非仅沿单一思路链前进。

## 核心要点
1. 在每一步生成多个候选推理步骤（分支），而非单一 continue
2. 使用 BFS 或 DFS 搜索策略探索推理树
3. 通过评估函数对分支进行评分和剪枝
4. TreeThink 是 ToT 思想的推广和系统化

## 代表工作
- [[TreeThink]]: 将 ToT 思想推广为通用模块化框架
- Yao et al. (2023): 提出 Tree-of-Thought 方法

## 相关概念
- [[CoT|Chain-of-Thought]]
- [[蒙特卡洛树搜索|MCTS]]
