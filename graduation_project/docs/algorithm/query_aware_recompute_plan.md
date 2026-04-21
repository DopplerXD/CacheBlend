# Query-aware 选择性重算方案（Chunk KV Cache 主线）

## 1. 背景与目标

当前主线不再把缓存单位定义为完整 session prompt，也不依赖 stale prompt 触发“非前缀命中”。RAG 样本中的每一段 `text` 都是一个 chunk：系统先将 chunk 独立 prefill 到 chunk KV 池，后续请求再按 chunk 命中缓存并重建完整上下文。

Query-aware 策略的目标是：在已经复用 chunk KV 的基础上，只更新与当前 query 语义更相关的部分 chunk token，避免为了选择重算位置而先执行一次完整 full prefill。

## 2. 当前实现结论

当前仓库中的 Query-aware 路径是：

1. 对 `chunk_texts` 中每个 chunk 独立预热，得到本地 position 下的 chunk KV；
2. 推理时 prefill `prefix_text`，读取 chunk KV，并对 chunk key 做 RoPE 重定位；
3. 按 `prefix + chunks` 的全局顺序拼接 KV；
4. 使用 query token 与 chunk token 的输入 embedding 余弦相似度打分；
5. 按 `recomp_ratio` 选择 Top-K chunk token；
6. 将选中 token 按 packed 序列执行一次 prefill，并传入其全局 `position_ids`；
7. 将 packed prefill 得到的 KV scatter 回完整 chunk KV 的对应全局位置；
8. 在融合后的 past 上完整 prefill `suffix_text`，随后进入 decode。

核心实现位于：

- `engine/inference_engine.py::_generate_chunk_aware`
- `engine/inference_engine.py::_warm_missing_chunk_caches`
- `engine/inference_engine.py::_select_chunk_token_indices`
- `engine/inference_engine.py::_recompute_selected_chunk_tokens_packed`
- `cache/kv_cache.py`
- `cache/kv_fusion.py`
- `model/hf_model.py`

## 3. 三种实验路径

### 3.1 Full Prefill

`full_prefill` 不读取 chunk KV。系统直接将

```text
prefix_text + chunk_texts[] + suffix_text
```

拼成完整输入，一次性执行 prefill，再进入 decode。该路径是标准质量基线，能够让所有 token 在完整上下文中参与计算，但不会获得 chunk 缓存带来的加速。

### 3.2 Full Reuse

`full_reuse` 读取或预热每个 chunk 的 KV，并直接拼接复用：

1. prefill `prefix_text`；
2. 对每个 chunk KV 做 RoPE key 重定位；
3. 拼接 `prefix KV + chunk KV`；
4. 在拼接后的 past 上完整 prefill `suffix_text`；
5. decode。

RoPE 重定位是必要步骤。chunk 独立预热时 key 已经按本地位置旋转，而完整 prompt 中 chunk 的位置由 prefix 长度和前面 chunk 的长度决定。当前实现将 key 从本地位置变换到全局位置：

```text
K_global = RoPE(global_pos) * RoPE(local_pos)^-1 * K_local
```

value 不含 RoPE 位置旋转，直接复用。Full reuse 不重算 chunk token，因此速度最快，但它只保证位置编码一致，不保证独立 chunk KV 与完整上下文 prefill 完全等价。

### 3.3 Query-aware

`query_aware` 从 full reuse 的拼接 KV 出发，再更新一部分与 query 更相关的 chunk token。它不再依赖 `suffix_len` 强制保护查询区域，因为 `suffix_text` 会在 chunk KV 融合之后完整 prefill。

## 4. Token 打分

当前实现优先使用显式传入的 `query_text`：

```text
query_token_ids = encode_no_special(query_text)
```

候选 token 只来自 `chunk_texts`。`prefix_text` 和 `suffix_text` 不进入 Top-K 候选区间，原因是 prefix 通常是任务模板，suffix/query 会在融合后完整 prefill。

候选 token 和 query token 都通过模型输入 embedding 层获得向量表示，然后做 L2 归一化和余弦相似度：

```text
sim = normalize(chunk_emb) @ normalize(query_emb).T
score_i = max_j sim(i, j)
```

这里的分数表示“某个 chunk token 与 query 中任一 token 的最大语义相关性”。

## 5. Packed Prefill 与 KV Scatter

选中索引后，系统构造：

```text
selected_token_ids = chunk_token_ids[selected_chunk_positions]
selected_position_ids = selected_global_positions
```

随后执行：

```text
forward_tokens(
    selected_token_ids,
    past_key_values=None,
    position_ids=selected_position_ids,
)
```

这会得到只包含 selected token 的 `selected_past_key_values`。系统再以 full reuse 拼接出的完整 KV 为底座，将 selected KV 按全局位置 scatter 回对应位置，未选中位置继续复用 chunk KV。

该流程是纯 Transformers 原型中的近似更新。packed prefill 保留了原始全局位置编号，但 selected token 在这次前向中只能看到 packed 序列内部的上下文，不等价于底层稀疏 attention 在完整上下文中的精确重算。

## 6. 与旧 kv_diff/session 路径的差异

| 维度 | 旧 session / kv_diff 路径 | 当前 chunk QAW 路径 |
| --- | --- | --- |
| 缓存单位 | 完整 prompt 或 session 前缀 | 单个 chunk |
| 触发方式 | 旧 prompt 与新 prompt 非前缀匹配 | `chunk_texts` 非空且 `recompute_strategy="query_aware"` |
| 选择依据 | 新旧 V 差异或旧 prompt 对齐 | query 与 chunk token 的 embedding 相似度 |
| 是否需要先 full prefill 得到 new KV | kv_diff 需要 | 不需要 |
| query/suffix 处理 | 依赖尾部强制重算参数 | suffix/query 在融合后完整 prefill |
| KV 拼接关键点 | session KV 对齐 | chunk key 的 RoPE 重定位 |

## 7. 参数建议

1. `recomp_ratio`：控制 chunk token Top-K 比例。低比例更快，高比例通常更接近 full prefill 质量。
2. `query_text`：建议显式传入，避免系统模板和回答格式污染 query 表示。
3. `chunk_namespace`：建议按数据集或实验划分，避免不同任务的同 token chunk 意外共用。

## 8. 论文表述边界

第三章应把当前方法描述为“基于 chunk KV 缓存的查询感知选择性重算”。可以强调：

- 每段检索文本独立预热为 chunk KV；
- Full Prefill 不走缓存，是标准质量基线；
- Full Reuse 通过 RoPE 重定位和 KV 拼接直接复用 chunk；
- Query-aware 在 Full Reuse 的基础上 packed 重算高相关 chunk token 并 scatter 融合；
- 当前实现避免了 query-aware 路径中的完整 new KV 预计算；
- packed prefill 是原型系统中的近似更新，不应表述为严格等价于 full prefill。
