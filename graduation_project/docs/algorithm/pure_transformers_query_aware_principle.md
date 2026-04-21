# 纯 Transformers 下 Chunk Query-aware 选择性重算原理

## 1. 文档目的

本文档描述当前仓库中“纯 Transformers + HF causal LM”版本的 chunk 级 KV 缓存与 Query-aware 选择性重算实现，覆盖：

- chunk 预热与缓存池；
- RoPE key 重定位与 KV 拼接；
- full prefill、full reuse、query-aware 三条路径；
- packed prefill 与 KV scatter；
- 实验统计口径和方法边界。

对应核心代码：

- `schema/types.py`
- `cache/kv_cache.py`
- `cache/kv_fusion.py`
- `model/hf_model.py`
- `engine/inference_engine.py`

## 2. 请求结构

当前 chunk-aware 请求由以下字段组织：

- `prefix_text`：任务说明、系统模板或 few-shot 前导文本；
- `chunk_texts`：可缓存文本块，一段 `text` 对应一个 chunk；
- `suffix_text`：当前 query、回答格式和生成入口；
- `query_text`：query-aware 打分使用的语义查询；
- `recompute_strategy`：`none` 或 `query_aware`；
- `recomp_ratio`：query-aware Top-K 重算比例；
- `chunk_namespace`：chunk KV 缓存命名空间。

只要 `chunk_texts` 非 `None`，`InferenceEngine.generate()` 就进入 chunk-cache 主链路。旧 session-prefix 缓存和 `kv_diff` 兼容代码仍存在，但不是当前实验和论文主线。

## 3. Chunk 预热

每个 chunk 独立编码并单独执行 prefill：

```text
chunk_i tokens -> forward(past=None, position_ids=0..len_i-1) -> chunk_i KV
```

缓存 key 由命名空间、模型标识和 chunk token ids hash 共同决定。这样同一模型、同一命名空间、同一 token 序列的 chunk 可以跨样本复用。

预热得到的是局部 KV。它没有看到当前样本的 prefix、其他 chunk 和 suffix，因此不能直接当作完整 prompt KV 使用。后续拼接时必须先处理 RoPE 位置。

## 4. RoPE 重定位

Yi/Qwen/LLaMA 系模型的 key 在 attention 中会被 RoPE 按位置旋转。chunk 独立预热时，chunk 内第 `j` 个 token 使用的是本地位置 `j`；放入完整 prompt 后，它的真实位置可能是 `p+j`。因此 key 需要从本地位置变换到全局位置：

```text
K_global = RoPE(global_pos) * RoPE(local_pos)^-1 * K_local
```

value 不经过 RoPE 旋转，可以直接复用。当前实现由 `model/hf_model.py` 读取模型 rotary embedding 参数并完成 key 的 unapply/apply 操作，输出 dtype 会保持与原 KV 一致，避免 BF16 与 FP32 混用。

## 5. 三条推理路径

### 5.1 Full Prefill

`full_prefill` 不读 chunk KV。系统直接将

```text
prefix_text + chunk_texts[] + suffix_text
```

拼成完整 token 序列，执行标准 prefill 后进入 decode。该路径最接近普通推理流程，但每条样本都会重复处理所有 chunk。

### 5.2 Full Reuse

`full_reuse` 的流程是：

1. prefill `prefix_text`；
2. 读取或预热每个 chunk KV；
3. 将每个 chunk key 从本地 RoPE position 重定位到全局 position；
4. 拼接 `prefix KV + chunk_1 KV + ... + chunk_n KV`；
5. 在拼接后的 past 上完整 prefill `suffix_text`；
6. decode。

这条路径不重算 chunk token，因此 TTFT 通常最低。它解决了 chunk KV 拼接时的位置编码问题，但不会让独立 chunk 的上下文状态严格等价于完整 prompt prefill。

### 5.3 Query-aware

`query_aware` 在 full reuse 的 KV 底座上进行部分更新：

1. 将所有 chunk token 作为候选区间；
2. 用 `query_text` 与 chunk token embedding 计算 cosine 相似度；
3. 按 `recomp_ratio` 选择 Top-K chunk token；
4. 把选中 token 打包成 packed 序列，传入对应全局 `position_ids` 前向；
5. 将 selected KV scatter 回完整 KV 的全局位置；
6. 在融合后的 past 上完整 prefill `suffix_text`；
7. decode。

因为 `suffix_text` 会在融合后完整 prefill，当前 chunk 主线不再把 `suffix_len` 作为 query-aware 的主参数。

## 6. Token 打分与选择

候选 token 表示来自模型输入 embedding 层，query token 表示来自显式传入的 `query_text`。系统做 L2 归一化后计算相似度矩阵：

```text
sim = normalize(chunk_emb) @ normalize(query_emb).T
score_i = max_j sim(i, j)
```

`score_i` 表示第 `i` 个 chunk token 与 query 中任一 token 的最大语义相关性。Top-K 数量由 `recomp_ratio` 决定：

```text
topk_num = floor(chunk_token_count * recomp_ratio)
```

选择结果以 chunk 内位置和完整 prompt 的全局位置共同记录。chunk 内位置用于取 token，完整位置用于 packed prefill 的 `position_ids` 和后续 KV scatter。

## 7. Packed Prefill 与 KV Scatter

选中 token 后，系统构造：

```text
selected_token_ids = [chunk_token_ids[i] for i in selected_chunk_positions]
selected_position_ids = selected_global_positions
```

然后执行一次短序列前向：

```text
forward_tokens(
    selected_token_ids,
    past_key_values=None,
    position_ids=selected_position_ids,
)
```

得到的 `selected_past_key_values` 长度为选中 token 数。随后系统以 full reuse 拼接出的完整 KV 为底座，将 selected KV 写回对应全局位置，未选中位置继续保留复用 KV。

这种方式验证了“选中后更新”的工程链路，但它是近似方法。selected token 在 packed prefill 中不会访问未选 token 的完整上下文，因此不能把结果表述为与 full prefill 严格等价。

## 8. 复杂度与性能直觉

设 chunk token 总数为 `L`，选中数量为 `K`，query token 数为 `Q`，hidden size 为 `H`。主要开销包括：

1. 打分：约 `O(L * Q * H)`；
2. Top-K：由 `torch.topk` 完成；
3. packed prefill：处理 `K` 个 token；
4. scatter：逐层将 selected KV 写回完整位置；
5. suffix prefill：在融合 KV 后完整处理 query/suffix。

因此，QAW 的速度不只由 `K < L` 决定，还受到 suffix 长度、模型规模、KV 拼接和显存状态影响。合理预期通常是 `full_reuse` 最快，`full_prefill` 最稳定，`query_aware` 位于两者之间并体现质量/时延折中。

## 9. 实验统计口径

建议报告：

- `*_avg_ttft_s`
- `*_avg_total_s`
- QA 数据集 F1；
- SAMSum Rouge-L；
- `recomputed_tokens`
- `chunk_cache_hits` / `chunk_cache_misses`
- `qaw_true_recompute_count`

曲线实验应以 `recomp_ratio` 为主参数。旧的 `suffix_len` 曲线不再属于 chunk-cache 主线。

## 10. 术语对照

- chunk KV cache：以单段文本为单位缓存 KV；
- local position：chunk 独立预热时的位置；
- global position：chunk 放回完整 prompt 后的位置；
- RoPE rebase：将 key 从 local position 旋转到 global position；
- full reuse：直接拼接并复用 chunk KV；
- packed prefill：只对 selected token 组成的短序列前向；
- scatter fusion：将 selected KV 按全局位置写回完整 KV。
