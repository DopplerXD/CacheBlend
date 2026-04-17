# 模型信息总结

## 模型位置
- Qwen2.5-1.5B: `/root/models/Qwen2.5-1.5B/`
- Yi-6B: `/root/models/Yi-6B/`

## 默认精度

| 模型 | 默认精度 | 说明 |
|------|----------|------|
| Qwen2.5-1.5B | BF16 (bfloat16) | config.json: `"torch_dtype": "bfloat16"` |
| Yi-6B | BF16 (bfloat16) | config.json: `"torch_dtype": "bfloat16"` |

**两者均为全精度模型，未量化。**

---

## Qwen2.5-1.5B

| 属性 | 值 |
|------|-----|
| 参数量 | 1.54B |
| 非嵌入参数量 | 1.31B |
| 架构 | Qwen2ForCausalLM (transformers + RoPE + SwiGLU + RMSNorm) |
| 层数 | 28 |
| 注意力头数 | 12 Q + 2 KV (GQA) |
| 隐层维度 | 1536 |
| 中间层维度 | 8960 |
| 上下文长度 | 32,768 tokens |
| 词表大小 | 151,936 |
| 训练语料 | 英文为主，多语言支持 |
| 许可 | Apache 2.0 |

---

## Yi-6B

| 属性 | 值 |
|------|-----|
| 参数量 | 6B |
| 架构 | LlamaForCausalLM (Llama架构) |
| 层数 | 32 |
| 注意力头数 | 32 Q + 4 KV (GQA) |
| 隐层维度 | 4096 |
| 中间层维度 | 11008 |
| 上下文长度 | 4K (可扩展至32K) |
| 词表大小 | 64,000 |
| 训练语料 | 3T tokens，双语(中英文) |
| 训练数据截止 | 2023年6月 |
| 许可 | Apache 2.0 |

---

## 验证脚本

| 脚本 | 路径 |
|------|------|
| Qwen2.5-1.5B 验证 | `/root/github/CacheBlend/verify_qwen.py` |
| Yi-6B 验证 | `/root/github/CacheBlend/verify_yi.py` |

运行方式:
```bash
source /root/github/CacheBlend/setup_env.sh && python <脚本路径>
```

---

## 显存需求 (FP16/BF16)

| 模型 | 最小显存 | 推荐GPU |
|------|----------|---------|
| Qwen2.5-1.5B | ~3GB | RTX 3060 (12GB) |
| Yi-6B | ~15GB | RTX 3090/4090 (24GB), A10, A30 |

---

## 硬件配置 (setup_env.sh)

```bash
conda activate yi
source /root/yi-query-aware/venv/bin/activate
```

环境: Python 3.10, torch, transformers>=4.38.0, accelerate, sentencepiece