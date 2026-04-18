# MVP 代码链路文档（按 `example/blend.py`）

本文描述当前 MVP 版本的实际执行链路，按 `example/blend.py` 的流程展开。

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

`example/blend.py` 会按样本循环，分别执行三种方法对比：
1. `kv_diff` 选择性重算
2. `query_aware` 选择性重算
3. `full_prefill` 基线

---

## 3. 阶段 1：按 chunk 逐步暖缓存
对应代码：`example/blend.py` 的 `warm_docs` 循环。

流程：
1. 每次追加一个文档块构建 `warm_prompt`
2. 调用 `engine.generate(..., use_cache=True, max_new_tokens=0)`
3. 引擎执行纯 prefill，并将本次 `prompt` 对应的 `past_key_values` 写回会话缓存

效果：
- 同一个 `session_id` 下，KV 缓存逐步积累，形成可复用前缀。

---

## 4. 阶段 2：KV-diff 重算路径
对应代码：`kvd_req`。

流程：
1. 构造完整问答 prompt（全部文档 + query）
2. `engine.generate(use_cache=True)` 读取会话缓存
3. 若缓存 token 是新 prompt 的前缀：
   - 复用已有 `past_key_values`
   - 只对增量 token 做 prefill
4. 进入 decode 循环逐 token 生成
5. 更新后的 KV 再写回缓存

关键指标：
- `reused_prefix_tokens`
- `first_token_latency_s`
- `total_latency_s`

---

## 5. 阶段 3：Query-aware 重算路径
对应代码：`qaw_req`。

流程：
1. 同样先读取 warm 缓存
2. 当 prompt 与缓存前缀不完全一致时，按 Query-Token 相关性选重算 token
3. 融合 old/new KV 后继续 decode

---

## 6. 阶段 4：基线路径生成（Normal Full Prefill）
对应代码：`base_req`。

流程：
1. 使用独立 `session_id` 且 `use_cache=False`
2. 引擎不读取/不写入 KV 缓存
3. 从零开始完整 prefill，再 decode

用途：
- 与阶段 2 做延迟和输出对比。

---

## 7. `InferenceEngine.generate` 内部主链路
入口文件：`engine/inference_engine.py`

1. `tokenizer.encode(prompt)` 得到 `prompt_token_ids`
2. 若 `use_cache=True`，尝试从 `KVCacheManager.get(session_id)` 读取会话 KV
3. 前缀命中则增量 prefill；否则全量 prefill
4. 调 `HFModelRunner.forward_tokens(...)` 执行 prefill
5. decode 循环：
   - 取最后 logits 采样下一个 token
   - 以 `next_token + past_key_values` 继续前向
6. 若 `use_cache=True`，将最新 `token_ids + past_key_values` 写回 `KVCacheManager.put(...)`
7. 返回 `GenerateResult`

---

## 8. KV 缓存模块行为（LRU）
入口文件：`cache/kv_cache.py`

1. `OrderedDict` 维护会话顺序
2. `get` 命中后 `move_to_end`（更新最近使用）
3. `put` 后若超过 `max_sessions`，淘汰最久未使用会话
4. 每次访问会先做 TTL 过期清理

---

## 9. 后续改造点（query-aware token selection）
预留位置：`InferenceEngine._select_recompute_indices`

当前该函数返回“全量索引”，未启用部分重算。
后续可在该函数中接入：
1. Query-Token 相关性打分
2. Top-K 选择
3. 仅重算被选 token 的策略流程
