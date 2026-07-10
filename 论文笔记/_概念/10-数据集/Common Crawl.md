---
type: concept
aliases: [CommonCrawl, CC]
---

# Common Crawl

## 定义
Common Crawl 是一个大规模网络爬虫语料库，包含数十亿网页的原始 HTML 数据、元数据和文本提取，是训练大型语言模型最常用的开源预训练数据源之一。

## 核心要点
1. 每月更新，数据量达 PB 级别
2. 包含多种语言和领域的网页内容
3. 需经过清洗、去重、过滤等预处理才能用于训练
4. DeepSeekMath 团队从 CC 中提取了 120B tokens 的高质量数学语料

## 代表工作
- [[DeepSeekMath]]: 从 CC 中迭代提取数学网页构建 DeepSeekMath Corpus
- GPT-3, PaLM, LLaMA 等模型均使用 CC 作为预训练数据源

## 相关概念
- [[OpenWebMath]]
- [[DeepSeekMath Corpus]]