# CacheBlend MVP（Transformers + HF Causal LM + 内存KV缓存）

本分支已移除 `vLLM/PagedAttention` 相关实现，目标是先跑通一个最小可用版本：

1. 基于 `transformers` 加载本地 HF causal LM
2. 自实现会话级 KV 缓存（内存 + LRU + TTL）
3. 支持 prefill/decode 主链路
4. 预留后续 `query-aware token selection recompute` 扩展位

## 目录结构

```text
app.py                      # 单次请求 CLI 入口
config.py                   # 运行配置
model/hf_model.py           # 模型封装
cache/kv_cache.py           # 简单 LRU KV 缓存
engine/inference_engine.py  # 主链路引擎
schema/types.py             # 请求/响应结构
utils/logging_utils.py      # 日志工具
example/blend.py            # 按原 blend 链路的对比示例
inputs/*.json               # 示例输入数据
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速运行

### 1) 单次请求

```bash
python app.py --prompt "介绍一下CacheBlend的思路" --session-id demo
```

启用“高 KV 偏差 token 重算”实验路径：

```bash
python app.py \
  --prompt "介绍一下CacheBlend的思路" \
  --session-id demo \
  --recompute-strategy kv_diff \
  --recomp-ratio 0.16 \
  --suffix-len 32
```

启用“Query-aware 选择性重算”实验路径：

```bash
python app.py \
  --prompt "介绍一下CacheBlend的思路" \
  --query-text "CacheBlend的核心机制是什么" \
  --session-id demo \
  --recompute-strategy query_aware \
  --recomp-ratio 0.16 \
  --suffix-len 32
```

### 2) 示例链路（缓存路径 vs 全量prefill）

```bash
python example/blend.py
```

## 说明

- 当前版本只做 MVP，不含并发调度、分页缓存、生产级容错。
- KV 缓存为进程内内存缓存；重启进程后缓存会丢失。
- 已支持两种实验策略：`kv_diff` 与 `query_aware`（`engine/inference_engine.py`）。
