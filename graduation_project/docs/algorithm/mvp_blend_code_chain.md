# MVP 代码链路文档（按 `example/blend.py` 与曲线实验脚本）

本文描述当前 MVP 版本的实际执行链路，重点说明 full prefill、full reuse、kv_diff 与 query-aware 的触发方式。该文档用于支撑论文第三章的方法落地说明，也可服务第四章实验设计。

## 1. 启动与组件初始化
入口：`example/blend.py`

1. 加载配置 `RuntimeConfig`（`config.py`）
2. 初始化日志（`utils/logging_utils.py`）
3. 加载模型封装 `HFModelRunner`（`model/hf_model.py`）
4. 初始化 KV 缓存管理器 `KVCacheManager`（`cache/kv_cache.py`）
5. 组装推理引擎 `InferenceEngine`（`engine/inference_engine.py`）

---

## 2. 样本处理循环（`inputs/1.json` 到 `inputs/10.json`）
每个样本包含：
- `chunk_num`
- 若干文档块 `0..chunk_num-1`
- `query`

`example/blend.py` 会按样本循环，执行四类路径：
1. `full_prefill`：关闭缓存，从零完整 prefill；
2. `full_reuse`：先对同一 prompt warm cache，再完整复用缓存；
3. `kv_diff`：从非前缀旧缓存启动，按 KV 偏差选择更新 token；
4. `query_aware`：从同一份非前缀旧缓存启动，按 query-token 相似度选择更新 token。

---

## 3. 阶段 1：构造 final prompt 与 stale prompt

流程：
1. `final_prompt`：文档内容 + 真实 query；
2. `stale_prompt`：文档内容 + 占位问题 + 真实 query 片段；
3. `stale_prompt` 被用于提前写入旧缓存，使后续 `final_prompt` 与缓存形成“非前缀匹配”。

效果：
- `full_reuse` 使用同一 `final_prompt` 作为缓存来源，用于测量完整复用；
- `kv_diff` 和 `query_aware` 使用同一份 `stale_prompt` 缓存作为旧 KV 底座，用于公平比较两种选择性重算策略。

---

## 4. 阶段 2：Full Prefill 基线路径
对应代码：`base_req`。

流程：
1. 使用独立 `session_id`；
2. 设置 `use_cache=False`；
3. 引擎从零对完整 `final_prompt` 执行 prefill；
4. 进入 decode。

用途：
- 作为质量和时延的主要参照基线。

---

## 5. 阶段 3：Full Reuse 路径
对应代码：`reuse_template_id` 与 `reuse_req`。

流程：
1. 先调用 `warm_prompt_cache` 对 `final_prompt` 执行 `max_new_tokens=0` 的缓存写入；
2. 将模板缓存复制到新的 `session_id`；
3. 使用相同 `final_prompt` 发起请求；
4. 因为缓存 token 是新 prompt 的完整前缀，进入 `cache_prefix_reuse` 分支。

用途：
- 展示理想前缀复用的低时延；
- 不作为当前方法的主要质量基线，因为它依赖 prompt 完全一致。

---

## 6. 阶段 4：KV-diff 重算路径
对应代码：`kvd_req`。

流程：
1. 先用 `stale_prompt` 写入旧缓存；
2. 将同一份旧缓存复制给 `kvd_session_id`；
3. 用 `final_prompt` 发起请求，并设置 `recompute_strategy="kv_diff"`；
4. 因为旧缓存不是新 prompt 的前缀，进入 `_prefill_with_kv_diff_recompute`；
5. 该路径先 full prefill 得到 `new_past_key_values`，再用新旧 V 的 L2 差异选择 token；
6. 以旧 KV 为底座，将选中位置替换为 new KV，补算最后 token logits 后 decode。

关键指标：
- `recompute_mode`
- `recomputed_tokens`
- `first_token_latency_s`
- `total_latency_s`

---

## 7. 阶段 5：Query-aware 重算路径
对应代码：`qaw_req`。

流程：
1. 使用与 `kv_diff` 相同的 `stale_prompt` 旧缓存；
2. 用 `final_prompt` 发起请求，并设置 `recompute_strategy="query_aware"`；
3. 因为旧缓存不是新 prompt 的前缀，进入 `_prefill_with_query_aware_recompute`；
4. 根据 `query_text` 与候选 token embedding 的余弦相似度选择 token；
5. 对选中 token 执行 packed prefill；
6. 将 packed KV scatter 回旧 KV 底座；
7. 补算最后 token logits 后 decode。

用途：
- 这是论文第三章的核心方法路径。

---

## 8. `InferenceEngine.generate` 内部主链路
入口文件：`engine/inference_engine.py`

1. `tokenizer.encode(prompt)` 得到 `prompt_token_ids`
2. 若 `use_cache=True`，尝试从 `KVCacheManager.get(session_id)` 读取会话 KV
3. 前缀命中则进入 `cache_prefix_reuse`
4. 非前缀命中时，根据 `recompute_strategy` 进入 `kv_diff`、`query_aware` 或回退 full prefill
5. 调 `HFModelRunner.forward_tokens(...)` 执行 prefill 或补算 logits
6. decode 循环：
   - 取最后 logits 采样下一个 token
   - 以 `next_token + past_key_values` 继续前向
7. 若 `use_cache=True`，将最新 `token_ids + past_key_values` 写回 `KVCacheManager.put(...)`
8. 返回 `GenerateResult`

---

## 9. KV 缓存模块行为（LRU）
入口文件：`cache/kv_cache.py`

1. `OrderedDict` 维护会话顺序
2. `get` 命中后 `move_to_end`（更新最近使用）
3. `put` 后若超过 `max_sessions`，淘汰最久未使用会话
4. 每次访问会先做 TTL 过期清理

---

## 10. 曲线实验脚本链路

`example/blend_curve_common.py` 是 MusiQue、WikiMQA、CMRC 曲线实验的公共逻辑。

关键流程：
1. `run_fixed_baselines`：只运行一次 full prefill/full reuse 基线；
2. `evaluate_qaw_grid`：遍历 `recomp_ratio` 和 `suffix_len`；
3. 每个样本先用 `stale_prompt` 写缓存；
4. 每组参数复制同一份旧缓存，保证 query-aware 比较公平；
5. 输出 `run_summary`，后续由分析脚本汇总成 CSV 和图。

该链路对应论文第四章的实验部分，但第三章可以引用它说明 `recomp_ratio` 与 `suffix_len` 的工程含义。
