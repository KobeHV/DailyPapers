---
type: concept
aliases: [FIM, 填空训练]
---

# Fill-in-Middle

## 定义
一种训练数据构造策略，将文本分割为前缀-中间-后缀三段，训练模型根据前缀和后缀预测中间部分，增强模型的代码补全和文本填充能力。

## 核心要点
1. 在 DeepSeek-V3 中引入，V4 继承使用
2. 与 MTP 互补：FIM 关注上下文填充，MTP 关注未来预测
3. FIM 数据格式：(PRE, SUF, MID) 或 (SUF, PRE, MID) 随机选择
4. 对代码能力有明显提升

## 代表工作
- [[DeepSeek-V3]]: 预训练中使用 FIM
- [[DeepSeek-V4]]: 继承 V3 的 FIM 策略

## 相关概念
- [[Multi-Token Prediction]]
- [[Pre-Training]]
