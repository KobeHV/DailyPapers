---
type: concept
aliases: [DeepSeekMath语料库]
---

# DeepSeekMath Corpus

## 定义
DeepSeekMath Corpus 是 DeepSeekMath 团队从 Common Crawl 中通过迭代式数据筛选 pipeline 提取的高质量数学预训练语料库，共计 120B tokens，涵盖多语言（以英文和中文为主）数学内容。

## 核心要点
1. 规模是 OpenWebMath 的 9 倍、Minerva 数学语料的 7 倍
2. 使用 fastText 分类器进行 4 轮迭代筛选
3. 多语言覆盖，同时提升了英文和中文数学推理能力
4. 通过 10-gram 精确匹配去除基准污染

## 代表工作
- [[DeepSeekMath]]: 使用该语料训练 DeepSeekMath-Base 7B

## 相关概念
- [[Common Crawl]]
- [[OpenWebMath]]
- [[fastText]]