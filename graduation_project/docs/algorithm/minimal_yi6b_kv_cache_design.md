# Yi-6B Chunk KV Cache 最小实现设计

本文档描述当前项目的最小 KV 缓存设计。实现目标是用 HuggingFace Transformers 直接运行 Yi/Qwen/LLaMA 系 Causal LM，并在 RAG 场景中以 chunk 为粒度复用 KV。

## 1. 设计目标

当前目标：

1. 每段 `text` 独立预热为 chunk KV；
2. chunk KV 以进程内缓存池保存，支持 LRU 和 TTL；
3. 推理时按完整 Prompt 中的位置对 chunk key 做 RoPE 重定位；
4. 支持 `full_prefill`、`full_reuse`、`query_aware` 三条路径；
5. Query-aware 使用 packed Prefill 和 KV scatter 更新 selected chunk token；
6. 保持代码足够简单，便于实验和论文说明。

非目标：

- 不实现跨进程持久化 KV；
- 不实现 vLLM/PagedAttention 级调度；
- 不把当前 packed 重算表述为精确稀疏 Attention 内核；
- 不把完整 session 前缀缓存作为实验主线。

## 2. 模块划分

| 模块 | 职责 |
| --- | --- |
| `schema/types.py` | 定义 `GenerateRequest`、`GenerateResult` 和 chunk 统计字段 |
| `cache/kv_cache.py` | 管理 chunk KV 池与兼容缓存接口 |
| `cache/kv_fusion.py` | 完成 KV concat、truncate、pad 和 scatter |
| `model/hf_model.py` | 封装 tokenizer/model forward，提供 RoPE key 重定位能力 |
| `engine/inference_engine.py` | 编排 full prefill、full reuse、query-aware 主链路 |
| `example/*.py` | 数据集实验入口与指标汇总 |

## 3. 数据结构

### 3.1 Chunk KV 条目

每个缓存条目至少包含：

- chunk token ids；
- 每层的 key/value；
- token 长度；
- dtype/device 信息；
- 创建时间与最近访问时间；
- 命名空间与模型标识。

缓存 key 由以下信息组成：

```text
chunk_namespace + model_name_hash + token_ids_hash
```

这样可以避免不同模型或不同实验命名空间误用同一份 KV。

### 3.2 Chunk-aware 请求

当前请求由三段主体构成：

```text
prefix_text + chunk_texts[] + suffix_text
```

- `prefix_text`：任务说明或文档列表前导模板；
- `chunk_texts`：可缓存的 RAG 文档块或 few-shot 示例；
- `suffix_text`：当前 query、回答格式和生成入口；
- `query_text`：Query-aware 打分使用的语义查询文本。

只要 `chunk_texts` 非空，引擎就走 chunk-cache 路径。

## 4. RoPE 重定位

chunk 独立预热时，token 位置从 0 开始。放入完整 Prompt 后，chunk 的起始位置由 prefix 长度和前面 chunk 的长度决定。因此，拼接前必须将 key 从 local position 转到 global position：

```text
K_global = RoPE(global_pos) * RoPE(local_pos)^-1 * K_local
```

value 不经过 RoPE 旋转，可以直接复用。当前实现支持常见 Yi/Qwen/LLaMA 结构中的 `rotary_emb.inv_freq`，无法识别时应显式报错。

## 5. 三条推理路径

### 5.1 Full Prefill

不读取 chunk KV，直接对完整输入执行 prefill：

```text
prefix_text + chunk_texts[] + suffix_text
```

该路径作为标准质量基线。

### 5.2 Full Reuse

1. prefill `prefix_text`；
2. 读取或预热每个 chunk KV；
3. 对 chunk key 做 RoPE 重定位；
4. 拼接 prefix KV 与所有 chunk KV；
5. 在拼接后的 past 上完整 prefill `suffix_text`；
6. decode。

该路径不重算 chunk token，主要用于观察直接复用 chunk KV 的速度收益和质量变化。

### 5.3 Query-aware

1. 先构建 Full Reuse KV 底座；
2. 使用 `query_text` 与 chunk token embedding 计算 cosine 相似度；
3. 按 `recomp_ratio` 选择 Top-K chunk token；
4. 对 selected token 执行 packed Prefill，并传入全局 `position_ids`；
5. 将 selected KV scatter 回完整 KV；
6. 在融合后的 past 上完整 prefill `suffix_text`；
7. decode。

该路径避免了为了选择重算位置而先执行完整 new KV 预计算。

## 6. 最小可运行链路

```text
load model/tokenizer
-> build prefix_text, chunk_texts, suffix_text, query_text
-> warm chunk cache
-> run full_prefill
-> run full_reuse
-> run query_aware
-> collect TTFT / total latency / F1 or Rouge-L
```

## 7. 边界

- RoPE 重定位只解决位置编码一致性，不保证独立 chunk KV 与完整上下文 Prefill 严格等价。
- packed Prefill 保留全局位置编号，但 selected token 无法完整访问未选 token 的上下文。
- query/suffix 在 chunk KV 拼接或融合后完整 Prefill，因此不需要作为 chunk KV 预热的一部分。
- 当前缓存是进程内内存缓存，进程重启后会丢失。
