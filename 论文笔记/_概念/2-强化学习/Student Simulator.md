---
type: concept
aliases: [学生模拟器]
---

# Student Simulator

## 定义
使用 LLM 扮演学生角色，批量模拟人机辅导交互，用于评估 AI 导师的教学效果，降低人类实验成本。

## 数学形式
学生解题正确率：
$$r_{\text{sol}} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}[\text{student solves correctly on attempt } k]$$

## 核心要点
1. 常用的 Student LLM: Llama-3.1-8B-Instruct
2. 每次辅导后进行 $K=8$ 次独立解题尝试，取平均正确率
3. 需要过滤掉过于简单（正确率 >60%）或过难（正确率 <1%）的问题

## 代表工作
- [[PedagogicalRL-Thinking]]: 使用 Student Simulator 进行评估

## 相关概念
- [[Intelligent Tutoring System]]
- [[LLM Judge]]
