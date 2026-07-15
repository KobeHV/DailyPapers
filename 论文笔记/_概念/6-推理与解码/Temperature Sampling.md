---
type: concept
aliases: [温度采样, 温度调节, Temperature Parameter, Softmax Temperature]
---

# Temperature Sampling

## 定义
通过调整 softmax 函数中的温度参数 $T$ 来控制 LLM token 生成概率分布平滑程度的采样策略。$T \to 0$ 趋于确定性输出（贪心解码），$T$ 增大则增加输出的多样性和随机性。

## 数学形式

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

## 核心要点
1. $T=0$: 等价于 argmax（贪心解码），完全确定性
2. $T=1$: 标准 softmax，保持原始 logits 的比例关系
3. $T>1$: 分布趋于均匀，增加低概率 token 被选中的机会
4. 不同温度解决不同子集的问题——这是 [[TemperatureScaling]] 方法的核心发现
5. 温度是"免费"的超参数——不需要训练，仅在推理时调整

## 代表工作
- [[TemperatureScaling]]: 发现在 [[Test-Time Scaling]] 中不同温度互补，多温度采样可突破单温度性能天花板
- Self-Consistency (Wang et al., 2023): 使用非零温度进行多样化的 CoT 采样

## 相关概念
- [[Test-Time Scaling]]
- [[Test-Time Compute]]
- [[Softmax]]
- [[Pass@K]]
- [[Entropy]]
