---
type: concept
aliases: [尾风险评分, Tail-Risk Score]
---

# Tail-Risk Score

## 定义
TACO 中用于衡量每个生成 token 属于不可靠尾分布的风险评分，基于 token 的 surprisal 与局部熵期望 surprisal 的偏差。

## 数学形式
$$r^{\mathrm{tail}}_{i,t} = -\log p_{i,t} - H_{i,t} + \log \alpha$$

## 核心要点
1. 结合 token 级稀有度（$-\log p_{i,t}$）和局部不确定性（$H_{i,t}$）
2. 正值表示 token 处于不可靠尾分布中
3. $\alpha$ 控制识别严格度
4. 区分"有用的稀有探索"和"不可靠的尾 token"

## 代表工作
- [[TACO]]: 核心组件

## 相关概念
- [[TACO]]
- [[Positive-Credit Contamination]]
- [[Credit Assignment]]
