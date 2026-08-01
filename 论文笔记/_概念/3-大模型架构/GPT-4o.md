---
type: concept
aliases: [GPT-4 Omni]
---

# GPT-4o

## 定义
OpenAI 发布的多模态旗舰模型（"o" 代表 "omni"），支持文本、图像、音频的输入输出，在推理、效率和多语言能力上显著提升。

## 核心要点
1. 原生多模态：统一处理文本、视觉和音频，端到端训练
2. 推理能力优异，常用作 LLM-as-a-Judge 进行自动评估
3. 在数学推理、代码生成、安全对齐等基准上达到 SOTA
4. 被 [[Reward Granularity in RLVR]] 用作错误分析的自动标注器
5. 相比 GPT-4 Turbo 更快、更便宜、效果更好

## 代表工作
- [[Reward Granularity in RLVR]]: 使用 GPT-4o 作为 LLM-as-a-Judge 标注推理错误类型
- OpenAI (2024): "Hello GPT-4o" 官方发布

## 相关概念
- [[LLM-as-a-Judge]]
- [[Chain-of-Thought]]
- [[RLVR]]
