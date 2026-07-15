---
type: concept
aliases: [LiveCodeBench v6]
---

# LiveCodeBench

## 定义
一个动态更新的代码生成基准，从 LeetCode、AtCoder 等竞赛平台收集最新编程题目，避免数据污染问题。v6 版本包含 279 道题。

## 核心要点
1. 动态更新：题目来自最近的编程竞赛，训练集不会包含
2. 多样化的编程语言和难度级别
3. 自动评估：通过执行测试用例判断正确性
4. [[TemperatureScaling]] 中 Pass@1,024 为 25%-40%，难度高于 MATH500 但低于 AIME
5. 与 [[AIME]] 互补：AIME 测数学推理，LiveCodeBench 测代码推理

## 代表工作
- [[TemperatureScaling]]: LiveCodeBench v6 上多温度缩放提升 +4.0 到 +7.4 点

## 相关概念
- [[AIME]]
- [[MATH500]]
- [[Pass@K]]
