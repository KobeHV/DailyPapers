---
type: concept
aliases: []
---

# fastText

## 定义
fastText 是 Facebook AI Research 开发的高效文本分类和词向量训练库，使用词袋模型和 n-gram 特征进行快速文本分类，适合大规模文本分类任务。

## 核心要点
1. 使用词袋模型和子词 n-gram 特征
2. 训练速度快，适合大规模语料分类
3. DeepSeekMath 使用 fastText 作为数学网页分类器
4. 配置：向量维度 256，学习率 0.1，n-gram 最大长度 3，训练 3 轮

## 代表工作
- [[DeepSeekMath]]: 使用 fastText 从 Common Crawl 中筛选数学网页

## 相关概念
- [[Common Crawl]]
- [[DeepSeekMath Corpus]]