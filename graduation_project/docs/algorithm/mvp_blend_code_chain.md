# 当前 MVP 代码链路（Chunk KV Cache）

本文描述当前代码实际执行链路。旧版“stale prompt + session KV”触发方式已不再作为主线。

## 1. 初始化

实验脚本执行：

1. 初始化 `RuntimeConfig`。
2. 创建 `HFModelRunner`。
3. 创建 `KVCacheManager`。
4. 创建 `InferenceEngine`。

## 2. 样本构造

每个样本被拆成：

```text
prefix_text + chunk_texts[] + suffix_text
```

- QA 数据集：`chunk_texts` 来自 `ctxs[*]`。
- SAMSum：`chunk_texts` 来自 few-shot 示例。
- `suffix_text` 包含当前问题或待摘要对话。
- `query_text` 只用于 query-aware 打分。

## 3. 缓存预热

实验先调用一次 `max_new_tokens=0` 的 chunk-aware 请求。引擎会对每个 chunk 独立执行 prefill，并写入 chunk KV 池。

缓存 key：

```text
chunk_namespace + model_name_hash + token_ids_hash
```

## 4. Full Prefill

`use_cache=False` 时，引擎不读 chunk KV，直接对完整 token 序列执行一次 prefill，然后 decode。

该路径是质量和时延基线。

## 5. Full Reuse

`use_cache=True` 且 `recompute_strategy="none"` 时：

1. prefill `prefix_text`；
2. 读取每个 chunk 的独立 KV；
3. 将 chunk key 从本地 RoPE position 重定位到全局 position；
4. 按顺序拼接 prefix KV 与 chunk KV；
5. 在拼接后的 past 上完整 prefill `suffix_text`；
6. decode。

该路径不重算 chunk token。

## 6. Query-aware

`use_cache=True` 且 `recompute_strategy="query_aware"` 时：

1. 先按 full reuse 构建 prefix + chunks KV；
2. 用 `query_text` 与 chunk token embedding 做 cosine 相似度；
3. 按 `recomp_ratio` 选择 Top-K chunk token；
4. 对 selected token 使用全局 `position_ids` 做 packed prefill；
5. 将 selected KV scatter 回完整 KV；
6. 完整 prefill `suffix_text`；
7. decode。

当前主线不再使用 `suffix_len`，因为 query/suffix 已经完整 prefill。

## 7. 输出指标

逐样本输出包含：

- `chunk_cache_hits`
- `chunk_cache_misses`
- `recomputed_tokens`
- `recompute_mode`
- TTFT 与 total latency
- QA F1 或 SAMSum Rouge-L

汇总输出包含各方法平均时延、平均质量指标和 QAW 真重算样本数。
