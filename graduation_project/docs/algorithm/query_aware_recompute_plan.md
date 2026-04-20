# Query-aware 选择性重算方案（当前纯 Transformers 原型）

## 1. 背景与问题

`kv_diff` 策略需要先得到当前 prompt 的 `new KV`，再与 `old KV` 比较差异，因此在纯 Transformers 原型中会额外执行一次 full prefill。该策略适合作为数值差异对照，但不适合作为“避免完整重算”的主要方案。

Query-aware 策略的目标是：不先完整计算 `new KV`，而是直接利用当前 query 与候选 token 的语义相关性选择少量 token，对这些位置做近似更新。

## 2. 当前实现结论

当前仓库中的 Query-aware 路径已经不是“先 full prefill 再融合”的旧版本，而是：

1. 使用输入 embedding 计算 query-token 相似度；
2. 选择 Top-K token，并合并新增 token 与尾部强制保留 token；
3. 只把选中的 token 打包成短序列做一次 packed prefill；
4. 将 packed prefill 得到的 KV scatter 回旧 KV 底座；
5. 补算最后一个 prompt token 的 logits 后进入 decode。

核心实现位于 `engine/inference_engine.py`：

- `_compute_query_aware_scores`
- `_select_token_indices_from_scores`
- `_prefill_with_query_aware_recompute`
- `_build_blended_from_old_and_selected`

## 3. 触发条件

在 `InferenceEngine.generate()` 中，Query-aware 分支只有同时满足以下条件时触发：

1. `GenerateRequest.use_cache=True`
2. 当前 `session_id` 已存在旧缓存；
3. 旧缓存 token 序列不是新 prompt token 序列的前缀；
4. `GenerateRequest.recompute_strategy == "query_aware"`。

实验脚本通常先构造一个 `stale_prompt` 写入缓存，再把同一份旧缓存拷贝给最终请求，以制造“非前缀旧缓存”，从而稳定触发 query-aware 分支。

## 4. Token 打分

当前实现优先使用显式传入的 `query_text`：

```text
query_token_ids = encode_no_special(query_text)
```

若 `query_text` 为空，则回退到 prompt 尾部 `suffix_len` 个 token。

候选 token 和 query token 都通过模型输入 embedding 层获得向量表示，然后做 L2 归一化和余弦相似度：

```text
sim = normalize(cand_emb) @ normalize(query_emb).T
score_i = max_j sim(i, j)
```

这里的分数表示“候选 token 与 query 中任一 token 的最大语义相关性”。

## 5. Token 选择

通用选择器 `_select_token_indices_from_scores()` 负责构造最终索引集合。

默认 Query-aware 路径：

```text
S = TopK(score, recomp_ratio) union AddedTokens union ForcedSuffix
```

其中：

- `TopK(score, recomp_ratio)`：在候选区间中按分数选择高相关 token；
- `AddedTokens`：新 prompt 中超出旧缓存长度的新增 token；
- `ForcedSuffix`：prompt 尾部 `suffix_len` 个 token；
- 默认 `force_changed=False`，即不强制加入所有新旧内容不同的位置。

两个实验变体：

1. `qaw_variant="no_suffix"`：移除尾部强制保留；
2. `qaw_variant="random_topk"`：用随机 Top-K 替代 query-aware 分数选择。

## 6. Packed Prefill 与 KV Scatter

选中索引后，系统构造：

```text
selected_token_ids = [prompt_token_ids[i] for i in selected_indices]
selected_position_ids = selected_indices
```

随后执行：

```text
forward_tokens(
    selected_token_ids,
    past_key_values=None,
    position_ids=selected_position_ids,
)
```

这会得到只包含 selected token 的 `selected_past_key_values`。然后系统以旧 KV 为底座：

1. 若旧 KV 长度不足新 prompt 长度，则尾部补零；
2. 将 selected KV 按 `selected_indices` 写回完整序列位置；
3. 未选中位置继续复用旧 KV。

最后系统截取融合 KV 到 `new_len - 1`，用最后一个 prompt token 补算 logits，保证后续 decode 有正确入口。

## 7. 与 kv_diff 的差异

| 维度 | kv_diff | query_aware |
| --- | --- | --- |
| 选择依据 | 新旧 V 的 L2 差异 | query-token embedding 余弦相似度 |
| 是否需要先得到完整 new KV | 需要 | 不需要 |
| 是否强制 changed token | 是 | 否 |
| 主要用途 | 数值差异对照策略 | 本文核心方法 |
| 当前实现风险 | 额外 full prefill，显存峰值高 | packed prefill 近似，不完全等价 full prefill |

## 8. 参数建议

1. `recomp_ratio`：控制 Top-K 比例。低比例更快，高比例更接近 full prefill 质量。
2. `suffix_len`：保护 prompt 尾部 query 区域。常用值为 `24` 或 `32`。
3. `query_text`：建议显式传入，避免系统模板和文档说明语句污染 query 表示。
4. `qaw_variant`：只作为消融实验使用，正文主方法采用 `default`。

## 9. 论文表述边界

第三章中应把当前方法描述为“纯 Transformers 原型中的近似选择性重算”。它验证了 query-aware token selection 的可行性，但没有实现 vLLM/PagedAttention 级别的底层稀疏重算内核。

可以强调：

- 当前方法避免了 query-aware 路径中的完整 new KV 预计算；
- 当前方法通过 packed prefill 降低被更新 token 数量；
- 当前方法的质量和速度取决于 `recomp_ratio`、`suffix_len` 和任务类型；
- 当前方法不是严格等价于 full prefill 的数学精确重算。
