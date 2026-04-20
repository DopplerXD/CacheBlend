# CacheBlend 改为“基于 Query-Token 语义相关性”选择重算 Token（历史简要方案）

> 说明：本文档是早期面向 vLLM/CacheBlend attention backend 的改造设想。当前仓库的主线实现已经转为纯 Transformers 原型，核心代码在 `engine/inference_engine.py`。论文第三章应以 `pure_transformers_query_aware_principle.md` 和 `query_aware_recompute_plan.md` 为准，本文档只作为方法来源和历史对照。

## 目标
当前实现在检查层用新旧 `V` 的 L2 差异选重算 token：
- `vllm_blend/vllm/attention/backends/xformers.py:204-217`

你要改成：优先重算与 query token 语义最相关的 token（不依赖 chunk 边界）。

## 先确认现状
现有 CacheBlend 的选择粒度是 **token**，不是 chunk：
- 直接对 token 级 `temp_diff` 做 `topk`（`xformers.py:209-211`）
- 结果 `imp_indices` 也是 token 索引（`xformers.py:217`）

因此选择时与“这个 token 来自哪个 chunk”没有显式关系。

## 推荐做法（最小侵入）

### 1) 在检查层直接做 Query-Token 相似度打分
改 `xformers.py` 的 `status == 1` 分支（`204-217` 附近）：
- 现在是：`temp_diff -> topk`
- 改为：`query-token cosine similarity -> topk`

做法：
1. 用 `suffix_len` 划分 query 区域（尾部）和候选区域（前缀）。  
   - `query_tokens = query[base_len:total_len]`
   - `cand_tokens = query[:base_len]`
2. 对 `cand_tokens` 与 `query_tokens` 做归一化点积（cosine）。
3. 用 `max/mean` 聚合成每个候选 token 的相关性分数。
4. 对相关性分数取 `topk`，再拼上尾部强制保留 token。

示意（伪代码）：
```python
last_len = cache_fuse_metadata['suffix_len']
total_len = value.shape[0]
base_len = total_len - last_len

topk_num = int(base_len * cache_fuse_metadata['recomp_ratio'])
q_query = query[base_len:total_len]         # [Q, H, D]
q_cand = query[:base_len]                   # [T, H, D]

# 拉平 head 维后做 cosine（也可按 head 分开）
q_query_f = q_query.reshape(q_query.shape[0], -1)
q_cand_f = q_cand.reshape(q_cand.shape[0], -1)
q_query_f = torch.nn.functional.normalize(q_query_f, dim=-1)
q_cand_f = torch.nn.functional.normalize(q_cand_f, dim=-1)

# sim: [T, Q]，每个候选 token 与所有 query token 的相似度
sim = q_cand_f @ q_query_f.transpose(0, 1)
rel = sim.max(dim=1).values   # 或 sim.mean(dim=1).values
top_indices = torch.topk(rel, k=topk_num).indices

last_indices = torch.arange(base_len, total_len, device=value.device)
top_indices, _ = torch.sort(top_indices)
imp_indices = torch.cat([top_indices, last_indices])
cache_fuse_metadata['imp_indices'] = imp_indices
query = query[imp_indices]
```

### 2) 扩展 metadata 字段
在 `llama.py` 初始化 `cache_fuse_metadata` 时加字段（可选但建议）：
- `selector_mode: "query_token_semantic"`（或 `"kvd"/"hybrid"`）
- `rel_reduce: "max"`（或 `"mean"`）

位置：
- `vllm_blend/vllm/model_executor/models/llama.py:300-310`

### 3) 可选：混合 Query-Token 与 KV 偏差（更稳）
可避免纯语义策略漏掉内部状态变化大的 token：

- `score = alpha * norm(query_token_rel) + (1 - alpha) * norm(kv_diff_score)`
- 典型：`alpha = 0.6~0.8`

## 注意点
1. 该方法依赖“尾部 token 基本就是 query”这个前提，`suffix_len` 要正确。
2. 你当前 `blend.py` 里 `last_len = len([q_ids+s_end])` 实际恒为 1；建议改成真实 query+suffix token 数。
3. `max` 聚合更激进，`mean` 更平滑；建议都做 A/B。
4. 先离线验证：固定 `recomp_ratio`，比较 TTFT 与答案质量（EM/F1）。

## 一句话
最小改法是：在 `xformers.py` 检查层直接用 query token 与候选 token 的 cosine 相似度做 top-k，替换原先的 `kv_diff top-k`。

## 当前实现对应关系

早期设想中的几个概念，在当前纯 Transformers 原型中的对应关系如下：

| 早期 vLLM 设想 | 当前纯 Transformers 实现 |
| --- | --- |
| `query-token cosine similarity -> topk` | `_compute_query_aware_scores` + `_select_token_indices_from_scores` |
| `suffix_len` 尾部强制保留 | `GenerateRequest.suffix_len` 与 `ForcedSuffix` |
| `selector_mode` | `GenerateRequest.recompute_strategy="query_aware"` |
| `rel_reduce="max"` | 当前固定采用 `sim.max(dim=1).values` |
| `imp_indices` | `selected_indices` |
| 回填 old/new KV | `_build_blended_from_old_and_selected` |

当前实现没有修改 vLLM/xFormers 内核，而是在 `HFModelRunner.forward_tokens` 之上执行 selected token packed prefill，再将 KV scatter 回旧缓存底座。
