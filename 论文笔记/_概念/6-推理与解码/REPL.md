---
type: concept
aliases: [REPL, Read-Eval-Print Loop, 交互式编程环境]
---

# REPL (Read-Eval-Print Loop)

## 定义
一种交互式编程环境，读取用户输入、求值、打印结果、循环。在形式化证明中，REPL 服务器提供实时代码验证和状态查询。

## 核心要点
1. 在 TreeThink 中，REPL 验证器连接形式化语言的 REPL 服务器进行实时证明验证
2. 提供二元反馈（通过/不通过），用于指导搜索方向
3. 支持 Lean 4（Kimina server）和 Rocq/Coq 8.20（rocq-ml-server）
4. 是 TreeThink Termination 组件的核心验证机制

## 代表工作
- [[TreeThink]]: 集成 REPL 验证作为实时证明检查手段

## 相关概念
- [[Lean 4]]
- [[Coq|Rocq]]
