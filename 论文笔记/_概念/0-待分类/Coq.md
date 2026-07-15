---
type: concept
aliases: [Rocq, Coq, Coq 8.20]
---

# Rocq (Coq)

## 定义
Rocq（原名 Coq，2025 年更名）是一个交互式定理证明器，基于归纳构造演算（Calculus of Inductive Constructions），支持形式化证明的开发和验证。

## 核心要点
1. 基于高阶类型论的形式化验证平台
2. **rocq-ml-server**: Rocq 的 ML 服务器接口，支持外部系统的通信
3. Coq 8.20 是 TreeThink 当前支持的版本
4. 广泛应用于形式化验证（CompCert C 编译器验证等）

## 代表工作
- [[TreeThink]]: 支持通过 rocq-ml-server 与 Rocq 进行同步验证

## 相关概念
- [[Lean 4]]
- [[Isabelle]]
- [[REPL]]
