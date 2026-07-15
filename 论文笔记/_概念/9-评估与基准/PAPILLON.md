---
type: concept
aliases: [Privacy-preserving Delegation]
---

# PAPILLON

## 定义
PAPILLON 是一个隐私保护委派任务（Privacy-Preserving Delegation），要求 LM 在处理用户请求时识别并脱敏个人身份信息（PII），同时从知识库中检索相关信息来回答。该任务测试模型在隐私与效用之间平衡的能力。

## 任务结构
多模块 LM 程序，包含：
1. **PII 检测模块**: 识别输入中的个人身份信息
2. **信息脱敏模块**: 对识别的 PII 进行脱敏处理
3. **检索模块**: 从脱敏后的查询中检索相关信息
4. **回答生成模块**: 基于检索结果生成最终回答

## 评估指标
综合得分，平衡隐私保护质量（不泄露 PII）和信息检索准确性。

## 代表工作
- [[mmGRPO]]: Ziems et al., 2025. 使用 mmGRPO 优化 PAPILLON 程序

## 相关概念
- [[DSPy]]: PAPILLON 的实现框架
