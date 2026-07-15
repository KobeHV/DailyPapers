---
type: concept
aliases: [TTS, 测试时缩放, Test-Time Compute Scaling, Inference-Time Scaling]
---

# Test-Time Scaling

## 定义
在推理阶段通过增加计算量（更多采样、更长推理链、搜索等）来提升模型输出质量的技术范式，与在训练阶段投入计算（Training-Time Scaling）相对。

## 数学形式

$$\mathrm{Pass@}K = 1 - \frac{\binom{N-C}{K}}{\binom{N}{K}}$$

## 核心要点
1. TTS 通过生成多条推理轨迹并选择最佳结果来提升性能
2. 典型方法包括 [[Best-of-N]]、[[Majority Voting]]、Self-Consistency、树搜索（[[蒙特卡洛树搜索|MCTS]]）
3. 与 [[CoT|Chain-of-Thought]] 正交——CoT 提升单条轨迹质量，TTS 提升多轨迹选择质量
4. 存在性能饱和现象：单温度下 $K$ 增至一定规模后收益趋零
5. 温度维度的引入（Temperature Scaling）扩展了 TTS 的边界

## 代表工作
- [[TemperatureScaling]]: 首次系统研究温度维度在 TTS 中的作用，多温度采样突破单温度性能天花板
- [[Best-of-N]]: 最简单的 TTS 形式，生成 N 个样本选最佳
- Self-Consistency (Wang et al., 2023): 通过多数投票聚合多条 CoT 推理链

## 相关概念
- [[Test-Time Compute]]
- [[Temperature Sampling]]
- [[Pass@K]]
- [[Best-of-N]]
- [[CoT]]
