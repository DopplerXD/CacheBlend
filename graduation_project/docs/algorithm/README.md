# 第三章算法文档阅读与写作索引

本文档面向本科毕业论文第三章“基于查询感知的 KV 选择性重算方法”，用于整理 `graduation_project/docs/algorithm/` 下各文档的用途、论文写作关注点，以及当前仓库中需要对应阅读的代码。

## 1. 与论文材料的对应关系

结合论文大纲、任务书和开题报告，第三章需要回答的问题是：

1. 为什么在长序列 RAG 场景中需要选择性 KV 重算；
2. 为什么 Token 选择不应只依赖固定比例或数值差异，而应引入查询语义；
3. 如何用 query-token 相似度为候选 Token 打分；
4. 如何把选中的 Token 转化为可运行的 KV 更新流程；
5. 当前原型系统的实现边界、复杂度和局限是什么。

任务书强调的是“查询感知 Token 选择与缓存更新策略”；开题报告强调的是“embedding × query 相似度、关键 Token 稀疏性、推理成本节省”；论文大纲要求第三章覆盖“问题定义、总体思路、Token 选择、选择性重算、复杂度分析”。

## 2. 文档优先级

### 必须重点阅读

1. `pure_transformers_query_aware_principle.md`
   - 第三章的主依据。
   - 覆盖触发条件、query-aware 打分、Top-K 选择、packed prefill、KV scatter、复杂度和局限。

2. `query_aware_recompute_plan.md`
   - 当前方案的简版说明。
   - 适合整理成 3.2 方法总体思路和 3.4 选择性重算机制。

3. `mvp_blend_code_chain.md`
   - 当前实验脚本与推理链路说明。
   - 用于写清楚方法如何在原型系统中触发，以及 full prefill、full reuse、kv_diff、query_aware 的比较口径。

### 作为背景或对照阅读

4. `hkvd_token_recompute_impl.md`
   - 用于理解 kv_diff/HKVD 对照策略。
   - 第三章中只需简要比较：“kv_diff 基于新旧 V 差异，query-aware 基于查询语义相关性”。

5. `minimal_yi6b_kv_cache_design.md`
   - 用于理解为什么当前系统选择纯 Transformers + 会话级 KV Cache 的 MVP 路线。
   - 更适合第四章或系统实现章节，第三章只引用其设计边界即可。

6. `semantic_relevance_recompute_brief.md`
   - 历史 vLLM 改造设想，不是当前最终实现。
   - 只适合作为早期方案或方法来源，不应作为当前代码事实直接引用。

## 3. 第三章需要关注的核心信息

### 3.1 问题定义

应写清楚三种模式：

1. `full_prefill`：完整重算新 Prompt，质量最稳定，但 Prefill 开销最高；
2. `full_reuse`：完全复用已有 KV，开销最低，但上下文变化时可能不可靠；
3. `selective recomputation`：以旧 KV 为底座，只更新部分 Token，追求速度和质量折中。

关键变量建议统一为：

- `P_old`：旧 Prompt；
- `P_new`：当前 Prompt；
- `K_old, V_old`：旧缓存；
- `S`：需要重算或更新的 Token 索引集合；
- `recomp_ratio`：候选区间 Top-K 比例；
- `suffix_len`：尾部强制更新长度。

### 3.2 方法总体思路

论文叙事应采用“两层流程”：

1. 理想目标流程：查询打分、选择 Token、稀疏更新、KV 融合、继续解码；
2. 当前原型流程：embedding 打分、Top-K 选点、selected token packed prefill、scatter 回旧 KV、补算最后 token logits。

这样可以避免把当前工程原型夸大成已经实现了底层稀疏 attention 内核。

### 3.3 查询感知 Token 选择

当前实现采用输入 embedding 的 cosine 相似度，不是 attention 分数，也不是中间层 hidden state。

核心公式可写为：

```text
sim(i, j) = cosine(e_i, u_j)
score_i = max_j sim(i, j)
```

其中 `e_i` 是候选 Token 的输入 embedding，`u_j` 是 query token 的输入 embedding。最终集合：

```text
S = TopK(score, recomp_ratio) union AddedTokens union ForcedSuffix
```

默认 query-aware 路径不强制加入 changed token；`kv_diff` 路径才会强制 changed token。

### 3.4 选择性重算机制

当前实现的关键步骤是：

1. 计算 `overlap_len = min(old_len, new_len, len(old_tokens))`；
2. 选择 `selected_indices`；
3. 构造 `selected_token_ids` 和 `selected_position_ids`；
4. 调用模型对 selected token 执行 packed prefill；
5. 以旧 KV 为底座，将 selected KV scatter 回原始位置；
6. 截取到 `new_len - 1`，补算最后一个 Prompt token 的 logits；
7. 进入标准 decode。

这一节必须强调“近似性”：packed selected token 在短序列中前向，虽然保留原始 position id，但不能完全复现完整上下文注意力，因此不是数学上等价于 full prefill。

### 3.5 复杂度分析

建议拆成两层：

1. 理想稀疏机制：重算数量约为 `rL`，理论上随 `r` 下降而降低开销；
2. 当前原型机制：总开销由 query 打分、Top-K、packed prefill、scatter、logits 补算共同组成。

当前实现的 query 打分复杂度可写为：

```text
O(L_c * m * d)
```

其中 `L_c` 是候选 Token 数，`m` 是 query token 数，`d` 是 embedding 维度。

## 4. 需要重点阅读的代码

### 核心算法

1. `engine/inference_engine.py`
   - `_compute_query_aware_scores`：embedding cosine 打分；
   - `_select_token_indices_from_scores`：Top-K、forced suffix、added token、random/no_suffix 变体；
   - `_prefill_with_query_aware_recompute`：query-aware 主流程；
   - `_build_blended_from_old_and_selected`：packed KV scatter 回填；
   - `generate`：触发条件和 full prefill/cache reuse/query-aware 的分支。

2. `model/hf_model.py`
   - `lookup_token_embeddings`：获取输入 embedding；
   - `forward_tokens`：Prefill/decode 共用的前向接口；
   - `truncate_past_key_values`：补算最后 token logits 前截断 KV。

3. `cache/kv_cache.py`
   - `SessionKV` 和 `KVCacheManager`：会话级 KV 的保存、读取、TTL/LRU 淘汰。

4. `schema/types.py`
   - `GenerateRequest`：`recompute_strategy`、`recomp_ratio`、`suffix_len`、`query_text`、`qaw_variant`；
   - `GenerateResult`：`recompute_mode`、`recomputed_tokens`、TTFT 和总时延。

### 实验触发与参数

1. `example/blend.py`
   - 小规模 demo 对比：`full_prefill`、`full_reuse`、`kv_diff`、`query_aware`。

2. `example/blend_curve_common.py`
   - MusiQue、WikiMQA、CMRC 的曲线实验公共逻辑；
   - 使用 stale prompt 构造非前缀旧缓存，强制触发 query-aware 分支。

3. `example/blend_curve.py`
   - `recomp_ratio` 曲线；

4. `example/blend_suffix_curve.py`
   - `suffix_len` 曲线；

5. `example/blend_ratio_suffix_curve.py`
   - ratio × suffix_len 网格实验。

### 可视化与结果分析

1. `scripts/visualize_query_aware_selection.py`
   - 生成 query-token similarity heatmap、score mask、annotated text；
   - 适合第三章放“选择机制示意图”。

2. `graduation_project/analyse/reanalysis_qaw_prefill_only/result_interpretation.md`
   - 更适合第四章实验分析，但第三章可引用其中的参数解释口径。

3. `outputs/query_aware_figs/`
   - 已生成的选择可视化示例图，可作为第三章方法图素材来源。

## 5. 第三章推荐写作顺序

1. 从 RAG 长上下文 Prefill 重复计算切入，定义 full prefill/full reuse/selective recomputation；
2. 引出已有 kv_diff/HKVD 思路的不足：它依赖新旧 KV 数值差异，通常需要先得到 new KV；
3. 提出 query-aware 思路：与当前查询更相关的 Token 更值得更新；
4. 给出 embedding cosine 打分公式和 Top-K + suffix 的选择规则；
5. 给出当前原型的 packed prefill + scatter 融合流程图；
6. 单独说明近似性和工程边界；
7. 做复杂度分析，并为第四章实验埋下 `recomp_ratio` 与 `suffix_len` 的参数解释。

## 6. 论文中应避免的表述

1. 不要说当前实现已经完成 vLLM/PagedAttention 级别的高效稀疏内核；
2. 不要说 query-aware 一定优于 full prefill，应表述为“在合理参数下实现速度与质量折中”；
3. 不要把 `full_reuse` 当成主要质量基线，正文主基线应以 `full_prefill` 为准；
4. 不要把 `semantic_relevance_recompute_brief.md` 中的 vLLM 路径当作当前仓库实际代码；
5. 不要把 query-aware 分数写成 attention 分数，当前实现是输入 embedding 相似度。

## 7. 当前文档状态

截至本次整理，第三章的主文档链路如下：

```text
README.md
  -> pure_transformers_query_aware_principle.md
  -> query_aware_recompute_plan.md
  -> mvp_blend_code_chain.md
  -> hkvd_token_recompute_impl.md
```

其中 `pure_transformers_query_aware_principle.md` 是最完整的算法说明，`README.md` 用来决定论文关注重点和代码阅读顺序。
