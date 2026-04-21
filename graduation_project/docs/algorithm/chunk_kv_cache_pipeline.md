# Chunk 级 KV 缓存与 Query-aware 融合全流程

## 1. 目标

当前实现把 RAG 场景中的缓存单位从“完整 session prompt”改为“文本 chunk”。一组样本中的每段 `text` 都独立 prefill 并写入 chunk KV 池；后续请求不再要求完整 prompt 前缀匹配，而是按 chunk 命中缓存并重建完整上下文 KV。

主线对比三种方法：

1. `full_prefill`：完整输入直接 prefill，不读 chunk 缓存。
2. `full_reuse`：读取并拼接 chunk KV，不重算 chunk token。
3. `query_aware`：先拼接 chunk KV，再选择与 query 相关的 chunk token 做 packed prefill，并 scatter 回完整 KV。

## 2. 请求结构

chunk-aware 请求由四部分组成：

```text
prefix_text + chunk_texts[] + suffix_text
```

- `prefix_text`：任务说明或系统模板，不作为 chunk 缓存主体。
- `chunk_texts`：RAG 文档块或 few-shot 示例块，每段独立缓存。
- `suffix_text`：query、问题模板、`Answer:` 或 `Summary:` 等尾部输入。
- `query_text`：QAW 打分使用的语义查询文本。

只要 `GenerateRequest.chunk_texts` 非 `None`，引擎就进入 chunk-cache 路径。

## 3. Chunk 预热

每个 chunk 单独编码为 token，并以本地 position 从 0 开始执行一次 prefill：

```text
chunk_i tokens -> forward(past=None) -> chunk_i KV
```

缓存 key 由 `chunk_namespace + model_name + token_ids hash` 组成。这样同一模型、同一数据集命名空间、同一 chunk token 序列可以跨样本复用。

## 4. RoPE 重定位与 KV 拼接

独立 chunk prefill 得到的 key 已经按本地 position 旋转。拼接到完整上下文时，需要把 key 从本地位置变换到全局位置：

```text
K_global = RoPE(global_pos) * RoPE(local_pos)^-1 * K_local
```

value 不含 RoPE 位置旋转，直接复用。当前实现支持 Yi/Qwen/LLaMA 系常见 `rotary_emb.inv_freq` 结构；无法识别 RoPE 参数时会报错，避免静默拼错 KV。

拼接顺序为：

```text
prefix KV + rebased chunk_1 KV + ... + rebased chunk_n KV
```

随后 `suffix_text` 会在拼接后的 past 上完整 prefill，因此新主线不再需要 `suffix_len` 保护 query。

## 5. Full Reuse

`full_reuse` 流程：

1. prefill `prefix_text`；
2. 读取或预热每个 chunk KV；
3. 对每个 chunk key 做 RoPE 重定位；
4. 拼接得到完整 chunk 上下文 KV；
5. 在该 past 上完整 prefill `suffix_text`；
6. 进入 decode。

该方法不重算 chunk token，只验证 chunk KV 直接拼接复用的速度与质量。

## 6. Query-aware

`query_aware` 在 full reuse 的 chunk KV 拼接结果上额外执行选择性更新：

1. 对所有 chunk token embedding 与 query token embedding 做 cosine 相似度；
2. 按 `recomp_ratio` 选择 Top-K chunk token；
3. 对 selected token 做 packed prefill，并使用其全局 `position_ids`；
4. 将 selected KV scatter 回完整 chunk KV 的对应全局位置；
5. 在融合后的 past 上完整 prefill `suffix_text`；
6. 进入 decode。

当前 QAW 仍是纯 Transformers 原型中的近似重算：selected token packed prefill 只能访问 packed 序列内部上下文，不等价于内核级稀疏 attention。但它避免了 query-aware 路径先做完整 new KV 预计算。

## 7. 实验口径

- QA 数据集：MusiQue、WikiMQA、CMRC，质量指标为 F1。
- SAMSum：质量指标为 Rouge-L。
- 主参数：`qaw_ratio` / `recomp_ratio`。
- `suffix_len` 已从 chunk-cache 主线移除；旧脚本参数仅作为兼容入口，不参与 QAW 选择。

## 8. 主要代码位置

- `schema/types.py`：chunk-aware 请求和 chunk 统计字段。
- `cache/kv_cache.py`：chunk KV 池。
- `cache/kv_fusion.py`：KV concat 与 scatter。
- `model/hf_model.py`：RoPE key 重定位。
- `engine/inference_engine.py`：full prefill、full reuse、query-aware 三条主链路。
- `example/blend_samsum.py`：SAMSum + Rouge-L 实验。
