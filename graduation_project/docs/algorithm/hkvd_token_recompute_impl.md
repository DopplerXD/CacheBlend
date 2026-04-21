# HKVD / kv_diff 对照说明（非当前主线）

本文档只保留 HKVD / `kv_diff` 作为历史对照材料。当前实验和论文主线已经切换为 chunk 级 KV 缓存与 Query-aware 选择性重算，不再以 `kv_diff` 作为主要方法。

## 1. 历史思路

HKVD / `kv_diff` 类方法关注的是“哪些位置的新旧 KV 差异更大”。典型做法是先得到当前输入的完整 new KV，再与已有 KV 逐位置比较，按差异大小选择需要更新的 token。

在纯 Transformers 原型中，这一路径的问题很直接：为了计算差异，系统必须先对当前完整 Prompt 执行一次 Full Prefill。这样虽然可以得到较明确的差异分数，但已经付出了完整重算的主要代价，因此不适合作为“避免 Full Prefill”的核心方案。

## 2. 与当前 Query-aware 的区别

| 维度 | HKVD / kv_diff | 当前 Query-aware |
| --- | --- | --- |
| 缓存单位 | 历史 KV 对照 | chunk KV |
| 选择依据 | 新旧 KV 或 V 的差异 | query 与 chunk token 的 embedding 相似度 |
| 是否需要先得到完整 new KV | 通常需要 | 不需要 |
| 是否需要 RoPE 重定位 | 取决于旧缓存形态 | chunk 拼接时必须需要 |
| 主要作用 | 历史对照或消融参考 | 当前论文主方法 |

当前 Query-aware 不先计算完整 new KV，而是直接基于 query 语义为 chunk token 打分：

```text
score_t = max_j cos(embedding(chunk_token_t), embedding(query_token_j))
```

随后选出 Top-K chunk token，执行 packed Prefill，并将 selected KV scatter 回完整 KV。该路径更符合当前“chunk 级缓存复用 + 查询感知更新”的目标。

## 3. 论文使用建议

第三章不建议展开 HKVD / `kv_diff` 的实现细节。若需要提及，可以作为对照说明：

- `kv_diff` 通过新旧 KV 差异选择 token；
- 它通常需要先得到完整 new KV；
- 因而在当前纯 Transformers 原型中不适合作为主要加速方案；
- 本文采用 Query-aware，是为了避免在选点前执行完整 Prefill。

实验章节如果保留历史结果，也应明确其为对照或历史实现，不与当前 chunk-cache 主线混用。
