# 算法文档阅读索引

当前主线已经从 session 级旧缓存改为 chunk 级 KV 缓存。论文和实验说明应优先引用 `chunk_kv_cache_pipeline.md`，旧的 session-prefix/stale prompt 文档只作为历史背景。

## 推荐阅读顺序

1. `chunk_kv_cache_pipeline.md`
   - 当前实现的主文档。
   - 覆盖 chunk 预热、RoPE 重定位、KV 拼接、full reuse、query-aware 融合和实验口径。

2. `pure_transformers_query_aware_principle.md`
   - 纯 Transformers 原型中的 chunk 预热、RoPE 重定位、query-aware token 选择、packed prefill 和 KV scatter。

3. `query_aware_recompute_plan.md`
   - QAW 选择性重算的简版说明。
   - 可作为 embedding cosine 打分与 Top-K 选择的论文材料。

4. `kernel_sparse_attention_recompute_workload.md`
   - 说明当前纯 Transformers 原型与真正内核级稀疏 attention 的边界。

5. `hkvd_token_recompute_impl.md`
   - HKVD/kv_diff 的历史对照材料。
   - 当前实验主线不再使用 kv_diff。

## 当前论文表述口径

- 缓存单位：一段 `text` 是一个 chunk。
- 主流程：独立 chunk prefill -> chunk KV cache -> RoPE 重定位 -> KV 拼接 -> 可选 QAW 重算 -> query/suffix prefill -> decode。
- 对比方法：`full_prefill`、`full_reuse`、`query_aware`。
- 主参数：`recomp_ratio`。
- `suffix_len` 不再作为主方法参数；query/suffix 会在融合后的 KV 上完整 prefill。
- SAMSum 使用 Rouge-L；QA 数据集使用 F1。

## 主要代码位置

- `schema/types.py`：`prefix_text`、`chunk_texts`、`suffix_text`、chunk 命中统计。
- `cache/kv_cache.py`：chunk KV 池与 legacy session 缓存兼容层。
- `cache/kv_fusion.py`：KV concat 与 selected KV scatter。
- `model/hf_model.py`：Yi/Qwen/LLaMA 系 RoPE key 重定位。
- `engine/inference_engine.py`：chunk-aware `full_prefill/full_reuse/query_aware`。
- `example/blend_curve_common.py`：MusiQue、WikiMQA、CMRC、SAMSum 曲线实验公共逻辑。

## 应避免的表述

- 不要再把当前实现描述为“会话级 KV 缓存”或“完整前缀缓存的特殊形态”。
- 不要说 QAW 依赖 stale prompt 触发；当前是基于 chunk KV 的直接复用与融合。
- 不要把 `suffix_len` 写成主参数。
- 不要声称当前实现等价于 vLLM/PagedAttention 级稀疏 attention 内核。
