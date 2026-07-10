---
type: concept
aliases: []
---

# OpenWebMath

## 定义
OpenWebMath 是一个高质量数学网页文本数据集，从 Common Crawl 中筛选产生，共计 13.6B tokens。它被 DeepSeekMath 用作数据收集 pipeline 的种子语料。

## 核心要点
1. 从 Common Crawl 中过滤数学内容得到
2. 被 DeepSeekMath 作为初始种子语料训练 fastText 分类器
3. 规模 13.6B tokens，远小于后续的 DeepSeekMath Corpus (120B)

## 代表工作
- [[DeepSeekMath]]: 以 OpenWebMath 为种子语料构建迭代式数据收集 pipeline

## 相关概念
- [[Common Crawl]]
- [[DeepSeekMath Corpus]]