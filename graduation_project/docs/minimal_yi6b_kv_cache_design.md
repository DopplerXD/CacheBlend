# 不依赖 vLLM/PagedAttention 的 Yi-6B + Transformer 最小化 KV 缓存方案设计

## 1. 目标与边界
### 1.1 目标
基于 `transformers + Yi-6B` 自行实现一个“仅内存 KV 缓存”的最小推理系统，支持：
1. 单机单卡（先不做分布式）
2. prefill + decode 两阶段推理
3. 会话级 KV 缓存复用（同一会话连续问答）
4. 可扩展为“按策略选择重算 token”（后续增强）

### 1.2 非目标
1. 不实现 vLLM 的 PagedAttention / block manager / 连续批调度
2. 不实现跨请求共享前缀缓存（先不做 prefix cache 索引）
3. 不实现复杂并发调度与抢占
4. 不实现多机多卡 TP/PP

---

## 2. 最小化文件清单（仅保留必需）
建议项目结构：

```text
minimal_yi_kv/
  README.md
  requirements.txt
  app.py                    # CLI/服务入口
  config.py                 # 模型与运行参数
  model/
    yi_model.py             # 封装 transformers Yi-6B 加载与 forward
  cache/
    kv_cache.py             # KV 数据结构与读写管理（核心）
  engine/
    inference_engine.py     # 主链路：prefill/decode + cache 调用
  schema/
    types.py                # Request/Response/SessionState 数据结构
  tests/
    test_kv_cache.py
    test_engine_smoke.py
```

如果要再压缩，`schema/types.py` 可合并进 `engine/inference_engine.py`。

---

## 3. 核心模块设计

## 3.1 `model/yi_model.py`
职责：
1. 加载 `AutoTokenizer` 与 `AutoModelForCausalLM`
2. 暴露 `forward(input_ids, attention_mask, past_key_values=None, use_cache=True)`
3. 返回 `logits` 与 `past_key_values`

实现要点：
1. 首版假设模型支持 HuggingFace 标准 `past_key_values`
2. `torch_dtype` 优先 `bfloat16`（按显卡能力回退 fp16）
3. 只保留 generation 必需配置（温度、top_p、max_new_tokens）

## 3.2 `cache/kv_cache.py`
职责：
1. 管理会话级 KV：`session_id -> SessionKV`
2. 提供 KV 追加、读取、清理、过期回收
3. 简单内存策略（LRU + TTL + 最大会话数）

建议数据结构：
1. `SessionKV`:
   - `past_key_values`: tuple(layer -> (k, v))
   - `token_ids`: 已缓存 token 序列
   - `last_access_ts`
2. `KVCacheManager`:
   - `store: Dict[str, SessionKV]`
   - `max_sessions`
   - `ttl_seconds`

关键 API：
1. `get(session_id) -> Optional[SessionKV]`
2. `put(session_id, session_kv)`
3. `append(session_id, new_token_ids, new_past_key_values)`
4. `clear(session_id)`
5. `evict_if_needed()`

## 3.3 `engine/inference_engine.py`
职责：
1. 接收请求并判断是否命中会话 KV
2. 跑 prefill（首次）或增量 decode（续写）
3. 将新 KV 回写缓存
4. 返回文本与统计信息（延迟、token 数）

---

## 4. 主链路（MVP）

### 4.1 首次请求（无缓存）
1. tokenizer 编码 prompt -> `input_ids`
2. 调 `model.forward(..., past_key_values=None, use_cache=True)` 进行 prefill
3. 保存返回的 `past_key_values` 到 `KVCacheManager`
4. 进入逐 token decode 循环（每步输入上一步 token + past）
5. 每步更新 `past_key_values`，最终回写会话缓存

### 4.2 同会话续写请求（有缓存）
1. 读取 `session_id` 对应缓存 KV 与历史 token
2. 对“新增 token”做增量 prefill（只喂新 token）
3. 更新 KV 后进入 decode 循环
4. 生成完成后回写 KV

### 4.3 清理与回收
1. 请求结束触发 `evict_if_needed`
2. 按 TTL/LRU 清理旧会话，防止显存/内存持续增长

---

## 5. 与 vLLM/PagedAttention 相比的取舍

## 5.1 优点
1. 代码量小，易读易改
2. 便于快速验证自定义策略（例如你要改 Query-Token 相关性重算）
3. 依赖少，调试简单

## 5.2 缺点
1. 吞吐和并发能力明显弱于 vLLM
2. 显存利用率与缓存管理效率较低
3. 缺少连续批处理、分页缓存、调度优化

---

## 6. 未来可插拔扩展点（为 CacheBlend 类策略预留）
在 `inference_engine.py` 里加选择器接口：
1. `select_recompute_indices(strategy, hidden_states, query_states, kv_delta, ...)`
2. `strategy="none"`（默认全量）
3. `strategy="query_token_semantic"`
4. `strategy="kv_diff"`

这样 MVP 先跑通，后续只替换选择器逻辑，不动主框架。

---

## 7. 工作量评估（单人）
假设工程师熟悉 PyTorch/Transformers。

### 7.1 MVP（可跑通、可复用会话 KV）
1. 基础骨架与配置：0.5 天
2. Yi-6B 模型封装：0.5 天
3. KV 管理器（内存 + TTL/LRU）：1.0 天
4. 主链路（prefill/decode + session）：1.0 天
5. CLI/简单 API 与日志：0.5 天
6. 单测与 smoke test：1.0 天

合计：**4.5 天 ~ 5.5 天**

### 7.2 工程化增强（建议）
1. 观测指标（TTFT、tokens/s、命中率）：0.5 天
2. 异常恢复与 OOM 保护：0.5 天
3. 更完整测试与压测脚本：1.0 天

再加：**2.0 天**

### 7.3 加入“Query-Token 相关性重算”策略
1. 选择器接口 + 策略实现：1.0 天
2. 质量/速度 A/B（参数扫描）：1.0~2.0 天

再加：**2.0 天 ~ 3.0 天**

---

## 8. 风险与注意事项
1. Yi-6B 不同仓库版本在 `past_key_values` 结构上可能有差异，需要先做一次兼容验证。
2. 长上下文会让会话 KV 快速膨胀，必须做配额与淘汰策略。
3. 如果后续要并发，需考虑会话锁与 CUDA stream 竞争。
4. 不使用 PagedAttention 时，大 prompt 下延迟会明显上升。

---

## 9. 一句话总结
这条路线可在约 **1 周内**完成一个可用 MVP：用 Transformers 原生 `past_key_values` 实现会话级内存 KV 缓存，主链路清晰、可快速验证自定义重算策略，但性能与并发能力会显著弱于 vLLM/PagedAttention。
