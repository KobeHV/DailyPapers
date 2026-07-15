---
type: concept
aliases: [vLLM, Virtual LLM]
---

# vLLM

## 定义
一个开源的高性能大语言模型推理引擎，通过 PagedAttention 和连续批处理等技术实现高效 LLM 推理。

## 核心要点
1. **PagedAttention**: 受操作系统虚拟内存启发的高效注意力 KV 缓存管理
2. **连续批处理**: 动态调整批次大小，最大化 GPU 利用率
3. 支持多种量化格式（GPTQ、AWQ、FP8 等）
4. 提供 OpenAI 兼容的 API 接口
5. **LoRA 适配器支持**: 可在运行时切换不同 LoRA 权重

## 代表工作
- [[TreeThink]]: 使用 vLLM 作为核心推理后端，支持本地服务器和外部 API 两种模式

## 相关概念
- [[LoRA]]
- [[SFT]]
