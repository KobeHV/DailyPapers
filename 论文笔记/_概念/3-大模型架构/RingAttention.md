---
type: concept
aliases: [Ring Attention, 环形注意力, 块状 Transformer]
---

# RingAttention

## 定义
一种通过将输入序列分块并在多个设备间形成环形通信模式来实现近无限上下文长度的注意力机制，每个设备只存储并计算其分配的块。

## 核心要点
1. 将序列分割为块并分配到多个 GPU
2. 通过环形通信传递 KV 块实现跨设备注意力
3. 可实现几乎无限的上下文长度（受设备数量和内存限制）
4. 计算复杂度仍为 $O(N^2)$，但内存被分片

## 代表工作
- RingAttention with Blockwise Transformers for Near-Infinite Context
- [[HiLS-Attention]]: 被引作为全注意力的扩展方法的代表

## 相关概念
- [[Chunk-wise Sparse Attention]]
