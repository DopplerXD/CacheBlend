# CacheBlend MVP（Transformers + Chunk KV Cache）

本分支主线已从 session 级完整 prompt KV 复用改为 chunk 级 KV 缓存复用。RAG 样本中的每段 `text` 会作为独立 chunk 预热缓存，推理时通过 RoPE 重定位把多个 chunk KV 拼接成完整上下文，再进入 full reuse 或 query-aware 选择性重算。

## 当前能力

1. 基于 `transformers` 加载本地 HF causal LM。
2. 提供进程内 chunk KV 池（LRU + TTL）。
3. 支持 Yi/Qwen/LLaMA 系 RoPE key 重定位。
4. 支持三条实验路径：
   - `full_prefill`：不读缓存，完整输入一次 prefill。
   - `full_reuse`：直接拼接 chunk KV，不做重算。
   - `query_aware`：按 query 与 chunk token 的 embedding 相似度选择部分 chunk token 重算并融合。
5. SAMSum 使用 Rouge-L，QA 数据集继续使用 F1。

## 目录结构

```text
app.py                      # 单次请求 CLI 入口（保留旧 prompt 兼容路径）
config.py                   # 运行配置
model/hf_model.py           # 模型封装与 RoPE KV 重定位
cache/kv_cache.py           # session 兼容缓存 + chunk KV 池
cache/kv_fusion.py          # KV 拼接与 scatter 融合
engine/inference_engine.py  # full_prefill / full_reuse / query_aware 主链路
schema/types.py             # 请求/响应结构
example/blend_*.py          # 数据集实验脚本
inputs/*.json               # 示例输入数据
```

## 安装

```bash
pip install -r requirements.txt
```

## 实验运行

```bash
python example/blend_musique.py --count 30 --qaw-ratio 0.7
python example/blend_musique.py --count 30 --qaw-ratio 0.6 --qaw-type window
python example/blend_wikimqa.py --count 10 --qaw-ratio 0.3
python example/blend_cmrc.py --model-name /root/models/Qwen2.5-1.5B --count 30 --qaw-ratio 0.7
python example/blend_samsum.py --count 100 --qaw-ratio 0.7
```

Ratio 曲线：

```bash
python example/blend_curve.py --dataset samsum --count 50 --qaw-ratio-min 0 --qaw-ratio-max 1 --qaw-ratio-step 0.05
```

## 说明

- `suffix_len` 不再是 chunk-cache QAW 主参数；query/suffix 会在 chunk KV 拼接和可选 QAW 融合之后完整 prefill。
- `--qaw-type packed` 是默认原实现；`--qaw-type window` 会把 QAW 选中的 token 扩展为左侧 16 token 的连续窗口重算。
- `kv_diff` 和旧 session-prefix 缓存路径仍保留为兼容代码，但不是当前实验主线。
- KV 缓存为进程内内存缓存；重启进程后缓存会丢失。
