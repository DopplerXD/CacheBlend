# Query-aware 选择性重算代码原理详解

本文只讲当前项目实际使用的主线：`chunk-aware + query-aware packed`。

不再展开代码中保留但当前主线没有使用的兼容路径、旧缓存对象、旧重算策略和备用重算方式。阅读本文时可以把当前实现理解成一条固定链路：

```text
prefix_text + chunk_texts + suffix_text
  -> chunk 文本分别编码
  -> chunk KV 读取或预热
  -> chunk KV 做 RoPE 位置重定位
  -> 拼接成 prefix + chunks 的 KV
  -> 用 query 对 chunk token 打分
  -> 选 Top-K token
  -> packed 方式只重算这些 token
  -> scatter 写回完整 KV
  -> suffix/query 完整 prefill
  -> decode 生成答案
```

本文覆盖的核心文件为：

- `schema/types.py`
- `cache/kv_cache.py`
- `cache/kv_fusion.py`
- `model/hf_model.py`
- `engine/inference_engine.py`
- `config.py`
- `utils/logging_utils.py`

---

## 1. 最小背景：这个方法到底在省什么

### 1.1 token

大模型不是直接处理整段字符串，而是先把文本切成 token，再把 token 转成整数编号。

例如：

```text
"北京是中国的首都" -> [1234, 5678, 90, 3456]
```

代码中常见的 `List[int]`，很多时候就是 token id 列表。

### 1.2 prefill 和 decode

一次生成通常分两步：

1. `prefill`：模型先完整读入 prompt，建立上下文状态。
2. `decode`：之后每次只输入刚生成的 1 个 token，预测下一个 token。

prompt 越长，prefill 越贵。当前 query-aware 主要优化的就是长上下文 prefill。

### 1.3 KV cache

Transformer 注意力中，每个 token 会产生 key 和 value。生成下一个 token 时，历史 token 的 key/value 可以复用，不必重复计算。这份历史 key/value 就叫 KV cache，在 HuggingFace 里通常叫 `past_key_values`。

一个简化结构如下：

```text
past_key_values = (
  layer0: (key_tensor, value_tensor, ...),
  layer1: (key_tensor, value_tensor, ...),
  ...
)
```

每层 key/value tensor 的常见形状是：

```text
[batch_size, num_heads, seq_len, head_dim]
```

其中 `seq_len` 是序列长度，也是本项目拼接和写回 KV 时最常操作的维度。

### 1.4 chunk KV 复用

当前项目把长上下文拆成多个 chunk。每个 chunk 可以单独 prefill 并缓存。之后如果另一个请求又使用同一个 chunk，就可以直接取出它的 KV。

但是 chunk 单独计算时位置从 0 开始；拼到完整 prompt 里时，它的位置可能从 prefix 后面开始。对于使用 RoPE 位置编码的模型，key 向量里含有位置信息，所以复用 chunk KV 前要做位置重定位。

### 1.5 query-aware 的核心动机

直接复用 chunk KV 速度快，但它不知道当前 query 关注什么。query-aware 的做法是：

1. 用 query 的 embedding 和 chunk token 的 embedding 计算相似度。
2. 找出最相关的一部分 chunk token。
3. 只重算这些 token 的 KV。
4. 把新 KV 写回完整 KV 的原位置。

这样比完整 prefill 便宜，又比完全不重算更贴近当前 query。

---

## 2. 当前主线的文件分工

```text
schema/types.py
  定义 GenerateRequest 和 GenerateResult

cache/kv_cache.py
  管理 chunk KV 缓存池

cache/kv_fusion.py
  负责 KV 拼接、克隆、写回

model/hf_model.py
  封装 tokenizer、模型 forward、RoPE 重定位、embedding 查询

engine/inference_engine.py
  组织完整 query-aware packed 流程

config.py
  提供模型、设备、缓存容量等运行配置

utils/logging_utils.py
  提供统一日志格式
```

当前主线最关键的入口是 `InferenceEngine.generate()`：

```python
def generate(self, req: GenerateRequest) -> GenerateResult:
    if req.chunk_texts is not None:
        return self._generate_chunk_aware(req)
```

只要 `req.chunk_texts` 不是 `None`，引擎就进入 chunk-aware 路径。query-aware packed 就是在这条路径里执行的。

---

## 3. 请求与响应结构：`schema/types.py`

### 3.1 `GenerateRequest`

源码位置：`schema/types.py:7-41`。

完整 dataclass 里有一些兼容字段。当前主线真正需要关注的是下面这些：

```python
@dataclass
class GenerateRequest:
    session_id: str
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    use_cache: bool = True
    recompute_strategy: str = "none"
    recomp_ratio: float = 0.16
    query_text: str = ""
    qaw_variant: str = "default"
    qaw_type: str = "packed"
    prefix_text: str = ""
    chunk_texts: Optional[List[str]] = None
    suffix_text: str = ""
    chunk_namespace: str = "default"
```

字段说明：

| 字段 | 类型 | 当前主线中的作用 |
|---|---:|---|
| `session_id` | `str` | 请求标识。构造请求必填；默认 query-aware packed 不依赖它做选择 |
| `prompt` | `str` | dataclass 必填字段；进入 chunk-aware 后主要使用 `prefix_text/chunk_texts/suffix_text` |
| `max_new_tokens` | `int` | decode 最多生成多少个新 token |
| `temperature` | `float` | 采样温度；`0.0` 表示贪心取最大概率 token |
| `top_p` | `float` | nucleus sampling 阈值；贪心时基本不发挥作用 |
| `use_cache` | `bool` | 是否使用 chunk KV 缓存；为 `False` 时直接完整 prefill |
| `recompute_strategy` | `str` | 设置为 `"query_aware"` 时启用 query-aware packed |
| `recomp_ratio` | `float` | 从 chunk token 中选择多少比例重算 |
| `query_text` | `str` | 显式 query；非空时用于 query-aware 打分 |
| `qaw_variant` | `str` | 打分方式；默认 `"default"` 会转成 embedding 最大相似度 |
| `qaw_type` | `str` | 当前主线使用 `"packed"` |
| `prefix_text` | `str` | 放在所有 chunk 前面的提示文本 |
| `chunk_texts` | `Optional[List[str]]` | 文本块列表；非 `None` 时进入 chunk-aware 主线 |
| `suffix_text` | `str` | 放在所有 chunk 后面的 query 或输出要求 |
| `chunk_namespace` | `str` | chunk 缓存命名空间，隔离不同来源的数据 |

当前主线的请求结构可以写成：

```python
GenerateRequest(
    session_id="case-001",
    prompt="",
    max_new_tokens=32,
    use_cache=True,
    recompute_strategy="query_aware",
    recomp_ratio=0.16,
    query_text="问题文本",
    qaw_variant="default",
    qaw_type="packed",
    prefix_text="任务说明",
    chunk_texts=["文本块1", "文本块2", "文本块3"],
    suffix_text="问题和回答格式要求",
    chunk_namespace="default",
)
```

### 3.2 `GenerateResult`

源码位置：`schema/types.py:44-62`。

```python
@dataclass
class GenerateResult:
    generated_text: str
    full_text: str
    reused_prefix_tokens: int
    prompt_tokens: int
    generated_tokens: int
    total_latency_s: float
    first_token_latency_s: Optional[float]
    recompute_mode: str = "full_prefill"
    recomputed_tokens: int = 0
    chunk_count: int = 0
    chunk_cache_hits: int = 0
    chunk_cache_misses: int = 0
```

字段说明：

| 字段 | 含义 |
|---|---|
| `generated_text` | 只包含模型新生成的文本 |
| `full_text` | prompt 加生成结果的完整文本 |
| `reused_prefix_tokens` | 当前 chunk-aware 路径中表示命中缓存的 chunk token 数 |
| `prompt_tokens` | 完整 prompt 的 token 数 |
| `generated_tokens` | 实际生成的新 token 数 |
| `total_latency_s` | 请求总耗时 |
| `first_token_latency_s` | 生成第一个新 token 前的耗时 |
| `recompute_mode` | 当前路径通常是 `chunk_query_aware_recompute` |
| `recomputed_tokens` | query-aware packed 实际重算的 token 数 |
| `chunk_count` | chunk 数量 |
| `chunk_cache_hits` | 命中的 chunk 数 |
| `chunk_cache_misses` | 未命中的 chunk 数 |

---

## 4. 运行配置与日志

### 4.1 `RuntimeConfig`

源码位置：`config.py:7-35`。

当前主线需要关心的配置：

| 字段 | 默认值 | 作用 |
|---|---|---|
| `model_name` | `/root/models/Yi-6B` | 模型路径 |
| `device` | `cuda` | 推理设备 |
| `model_dtype` | `bfloat16` | 模型权重精度 |
| `attn_implementation` | `auto` | Transformers attention 后端 |
| `max_new_tokens` | `16` | 默认最大生成 token 数 |
| `temperature` | `0.0` | 默认贪心生成 |
| `top_p` | `1.0` | 默认不做 top-p 截断 |
| `kv_max_chunks` | `2048` | chunk 缓存容量上限 |
| `kv_ttl_seconds` | `3600` | chunk 缓存过期时间 |
| `log_level` | `INFO` | 日志级别 |

这些配置来自环境变量。例如 `MODEL_NAME`、`DEVICE`、`KV_MAX_CHUNKS`。

### 4.2 `setup_logger`

源码位置：`utils/logging_utils.py:12-29`。

函数签名：

```python
def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger
```

入参：

- `name`：日志器名称。
- `level`：日志级别；为空时读环境变量 `LOG_LEVEL`。

出参：

- 配好格式的 `logging.Logger`。

行级逻辑：

- `19` 行：通过 `logging.getLogger(name)` 获取日志器。
- `20` 行：如果没有 handler，才添加新的 `StreamHandler`，避免重复输出。
- `21-24` 行：设置日志格式。
- `26-27` 行：解析日志级别。
- `28` 行：关闭向 root logger 传播。
- `29` 行：返回 logger。

---

## 5. chunk KV 缓存：`cache/kv_cache.py`

当前主线只需要理解 chunk 级缓存。它保存的是某个 chunk 单独 prefill 后得到的 `past_key_values`。

### 5.1 `ChunkKV`

源码位置：`cache/kv_cache.py:25-34`。

```python
@dataclass
class ChunkKV:
    cache_key: str
    namespace: str
    token_ids: List[int]
    past_key_values: Any
    last_access_ts: float
```

字段说明：

| 字段 | 作用 |
|---|---|
| `cache_key` | chunk 的唯一缓存键 |
| `namespace` | 命名空间，用来隔离不同任务或数据来源 |
| `token_ids` | chunk 文本编码后的 token id |
| `past_key_values` | chunk 单独 prefill 得到的 KV |
| `last_access_ts` | 最后访问时间，用于过期和 LRU |

### 5.2 `KVCacheManager.__init__`

源码位置：`cache/kv_cache.py:45-55`。

函数签名：

```python
def __init__(
    self,
    max_sessions: int,
    ttl_seconds: int,
    logger,
    max_chunks: Optional[int] = None,
) -> None
```

当前主线关心的入参：

- `ttl_seconds`：缓存过期时间。
- `logger`：日志器。
- `max_chunks`：chunk 缓存最大数量。

当前主线关心的内部状态：

```python
self._chunk_store: "OrderedDict[str, ChunkKV]" = OrderedDict()
```

`OrderedDict` 的作用是维护访问顺序，方便删除最久没有使用的 chunk。

### 5.3 `build_chunk_key(namespace, model_name, token_ids)`

源码位置：`cache/kv_cache.py:103-110`。

函数签名：

```python
@staticmethod
def build_chunk_key(namespace: str, model_name: str, token_ids: List[int]) -> str
```

入参：

- `namespace`：命名空间。
- `model_name`：模型名或模型路径。
- `token_ids`：chunk token。

出参：

- 字符串形式的缓存 key。

行级逻辑：

```python
payload = ",".join(str(t) for t in token_ids)
digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
model_digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
return f"{namespace}:{model_digest}:{digest}"
```

含义：

1. 把 token id 序列转成字符串。
2. 对 token 序列做 SHA256，保证相同文本块得到稳定 key。
3. 对模型名也做哈希，避免不同模型误用同一份 KV。
4. 加上 namespace，避免不同数据来源互相污染。

### 5.4 `_evict_expired_chunks()`

源码位置：`cache/kv_cache.py:112-121`。

作用：

- 遍历 `_chunk_store`。
- 删除超过 TTL 的 chunk。
- 输出日志。

这保证缓存不会无限保存旧 chunk。

### 5.5 `get_chunk(cache_key)`

源码位置：`cache/kv_cache.py:123-131`。

函数签名：

```python
def get_chunk(self, cache_key: str) -> Optional[ChunkKV]
```

入参：

- `cache_key`：chunk 缓存键。

出参：

- 命中时返回 `ChunkKV`。
- 未命中时返回 `None`。

流程：

1. 先调用 `_evict_expired_chunks()` 清理过期 chunk。
2. 从 `_chunk_store` 里查找 key。
3. 没找到就返回 `None`。
4. 找到后更新 `last_access_ts`。
5. 把该 key 移到 `OrderedDict` 末尾，表示最近访问。
6. 返回缓存项。

### 5.6 `put_chunk(cache_key, namespace, token_ids, past_key_values)`

源码位置：`cache/kv_cache.py:133-144`。

函数签名：

```python
def put_chunk(
    self,
    cache_key: str,
    namespace: str,
    token_ids: List[int],
    past_key_values: Any,
) -> None
```

入参：

- `cache_key`：缓存键。
- `namespace`：命名空间。
- `token_ids`：chunk token。
- `past_key_values`：chunk KV。

出参：

- 无返回值。

流程：

1. 构造 `ChunkKV`。
2. 写入 `_chunk_store`。
3. 移到末尾，表示最近使用。
4. 调用容量淘汰逻辑。

### 5.7 `clear_chunks(namespace=None)`

源码位置：`cache/kv_cache.py:151-163`。

函数签名：

```python
def clear_chunks(self, namespace: Optional[str] = None) -> None
```

入参：

- `namespace=None`：清空所有 chunk。
- `namespace="xxx"`：只清理该命名空间下的 chunk。

出参：

- 无返回值。

### 5.8 `stats()`

源码位置：`cache/kv_cache.py:169-178`。

返回缓存统计。当前主线主要看：

```python
{
    "chunks": len(self._chunk_store),
    "max_chunks": self.max_chunks,
    "ttl_seconds": self.ttl_seconds,
}
```

---

## 6. KV 拼接和写回：`cache/kv_fusion.py`

当前 query-aware packed 会用到两个最关键操作：

1. 把多段 KV 拼成一段完整 KV。
2. 把 selected token 的新 KV 写回完整 KV 的指定位置。

### 6.1 `concat_past_key_values(left, right)`

源码位置：`cache/kv_fusion.py:10-24`。

函数签名：

```python
def concat_past_key_values(left: Any, right: Any) -> Any
```

入参：

- `left`：左侧 KV。
- `right`：右侧 KV。

出参：

- 沿序列维拼接后的 KV。

行级逻辑：

- `12-13` 行：如果左侧为空，返回右侧 clone。
- `14-15` 行：如果右侧为空，返回左侧 clone。
- `18` 行：逐层遍历左右 KV。
- `19-20` 行：取出每层的 key 和 value。
- `21` 行：`torch.cat([left_k, right_k], dim=-2)` 拼接 key。
- `22` 行：同样拼接 value。
- `23` 行：保留层里除 key/value 外的其他元素。
- `24` 行：返回 tuple。

为什么是 `dim=-2`？因为 key/value 常见形状是：

```text
[batch, heads, seq_len, head_dim]
```

倒数第二维就是 `seq_len`。

### 6.2 `clone_past_key_values(past_key_values)`

源码位置：`cache/kv_fusion.py:27-35`。

函数签名：

```python
def clone_past_key_values(past_key_values: Any) -> Any
```

入参：

- 原始 KV。

出参：

- key/value tensor 都 clone 后的新 KV。

作用：

- 避免修改缓存池中的原始 KV。

### 6.3 `scatter_selected_past_key_values(base, selected, selected_indices)`

源码位置：`cache/kv_fusion.py:50-66`。

函数签名：

```python
def scatter_selected_past_key_values(
    base_past_key_values: Any,
    selected_past_key_values: Any,
    selected_indices: torch.Tensor,
) -> Any
```

入参：

- `base_past_key_values`：完整底座 KV。
- `selected_past_key_values`：只包含被选中 token 的新 KV。
- `selected_indices`：这些 token 在完整 prompt 中的全局位置。

出参：

- 写回后的 blended KV。

行级逻辑：

- `54-55` 行：如果没有 selected KV 或索引为空，直接返回 base。
- `57` 行：记录 selected token 数。
- `59-60` 行：逐层遍历 base 和 selected。
- `61` 行：clone base 的 key/value。
- `62` 行：取 selected 的 key/value。
- `63` 行：把 selected key 写回 base key 的指定序列位置。
- `64` 行：把 selected value 写回 base value 的指定序列位置。
- `65-66` 行：返回新的 KV。

核心赋值语句是：

```python
base_k[:, :, selected_indices, :] = selected_k[:, :, :selected_num, :]
base_v[:, :, selected_indices, :] = selected_v[:, :, :selected_num, :]
```

意思是：

```text
完整 KV 的若干位置 = 新重算出来的 KV
```

这就是 query-aware packed 融合的核心。

---

## 7. 模型封装：`model/hf_model.py`

`HFModelRunner` 把 tokenizer、模型 forward、RoPE 重定位、embedding 查询统一封装起来。

### 7.1 `_resolve_dtype(dtype_name)`

源码位置：`model/hf_model.py:14-22`。

函数签名：

```python
def _resolve_dtype(dtype_name: str) -> torch.dtype
```

入参：

- `dtype_name`：`"bfloat16"`、`"float16"`、`"float32"` 等字符串。

出参：

- 对应的 PyTorch dtype。

### 7.2 `HFModelRunner.__init__`

源码位置：`model/hf_model.py:28-88`。

函数签名：

```python
def __init__(
    self,
    model_name: str,
    device: str,
    model_dtype: str,
    logger,
    attn_implementation: Optional[str] = None,
)
```

入参：

- `model_name`：模型路径或名称。
- `device`：推理设备。
- `model_dtype`：模型权重精度。
- `logger`：日志器。
- `attn_implementation`：attention 后端。

初始化流程：

1. 检查设备。如果请求 CUDA 但不可用，就回退 CPU。
2. 解析 dtype。CPU 上使用 float32。
3. 加载 tokenizer。
4. 解析 attention 后端。
5. 加载 `AutoModelForCausalLM`。
6. 把模型移动到目标设备。
7. 设置 eval 模式。
8. 检查模型是否支持 `num_logits_to_keep`。
9. 如果 tokenizer 没有 pad token，则使用 eos token 作为 pad token。

### 7.3 `encode(text)`

源码位置：`model/hf_model.py:89-90`。

函数签名：

```python
def encode(self, text: str) -> List[int]
```

入参：

- 文本字符串。

出参：

- token id 列表。

它会添加模型需要的 special tokens。当前主线中，`prefix_text` 使用这个方法编码。

### 7.4 `encode_no_special(text)`

源码位置：`model/hf_model.py:92-93`。

函数签名：

```python
def encode_no_special(self, text: str) -> List[int]
```

入参：

- 文本字符串。

出参：

- 不带 special token 的 token id 列表。

当前主线中，chunk 文本、suffix 文本、query 文本都用这个方法。这样可以避免每个 chunk 都插入额外 special token。

### 7.5 `decode(token_ids)`

源码位置：`model/hf_model.py:95-96`。

函数签名：

```python
def decode(self, token_ids: List[int]) -> str
```

入参：

- token id 列表。

出参：

- 解码后的文本。

### 7.6 `eos_token_id()`

源码位置：`model/hf_model.py:98-99`。

返回 EOS token id。decode 循环中如果生成了 EOS，就提前停止。

### 7.7 RoPE 重定位相关函数

chunk 单独 prefill 时，它的 position 从 0 开始。拼进完整 prompt 后，它的位置要变成全局 position。RoPE 重定位就是把 key 从原位置旋转关系转换到新位置旋转关系。

#### `_find_rotary_inv_freq()`

源码位置：`model/hf_model.py:122-130`。

作用：

- 遍历模型模块，找到 RoPE 的 `inv_freq`。

出参：

- `torch.Tensor`。

如果找不到，会抛出错误，因为无法做位置重定位。

#### `_rotate_half(x)`

源码位置：`model/hf_model.py:132-137`。

RoPE 公式中的辅助函数。它把向量拆成两半并旋转：

```text
[x1, x2] -> [-x2, x1]
```

#### `_apply_rope_to_key(key, positions)`

源码位置：`model/hf_model.py:139-162`。

函数签名：

```python
def _apply_rope_to_key(
    self,
    key: torch.Tensor,
    positions: Sequence[int],
) -> torch.Tensor
```

作用：

- 给 key 应用指定 positions 的 RoPE 旋转。

核心流程：

1. 找到 `inv_freq`。
2. 根据 positions 计算 cos/sin。
3. 取 key 中参与 RoPE 的维度。
4. 应用旋转公式。
5. 如果 key 还有不参与 RoPE 的维度，则拼回去。

#### `_unapply_rope_from_key(key, positions)`

源码位置：`model/hf_model.py:164-187`。

作用：

- 从 key 中移除指定 positions 的 RoPE 旋转。

重定位时需要先移除旧位置，再应用新位置：

```text
本地位置 key
  -> 移除本地 RoPE
  -> 得到近似原始 key
  -> 应用全局 RoPE
  -> 得到全局位置 key
```

#### `rebase_past_key_values_positions(past_key_values, source_start, target_start)`

源码位置：`model/hf_model.py:189-212`。

函数签名：

```python
@torch.inference_mode()
def rebase_past_key_values_positions(
    self,
    past_key_values: Any,
    source_start: int,
    target_start: int,
) -> Any
```

入参：

- `past_key_values`：某个 chunk 的 KV。
- `source_start`：chunk 原来的起始位置，当前主线中通常是 0。
- `target_start`：chunk 在完整 prompt 中的新起始位置。

出参：

- 重定位后的 KV。

行级逻辑：

- `194-195` 行：空 KV 直接返回。
- `196` 行：读取 KV 长度。
- `197-198` 行：构造源位置列表和目标位置列表。
- `199-204` 行：如果位置没变，只 clone 返回。
- `206-211` 行：逐层处理：
  - 对 key 先移除源位置 RoPE。
  - 再应用目标位置 RoPE。
  - value 不含 RoPE 旋转，clone 即可。
- `212` 行：返回新 KV。

这是 chunk KV 能拼到完整 prompt 中的关键步骤。

### 7.8 `get_past_len(past_key_values)`

源码位置：`model/hf_model.py:214-221`。

函数签名：

```python
@staticmethod
def get_past_len(past_key_values: Any) -> int
```

入参：

- KV。

出参：

- KV 中保存的序列长度。

核心代码：

```python
return int(past_key_values[0][0].shape[-2])
```

意思是取第 0 层 key tensor 的序列维长度。

### 7.9 `truncate_past_key_values(past_key_values, target_len)`

源码位置：`model/hf_model.py:223-238`。

函数签名：

```python
@staticmethod
def truncate_past_key_values(
    past_key_values: Any,
    target_len: int,
) -> Any
```

当前主线中，它主要用于没有 suffix 时补算最后一个 prompt token 的 logits。

入参：

- `past_key_values`：原 KV。
- `target_len`：保留前多少个 token 的 KV。

出参：

- 裁剪后的 KV。

### 7.10 `lookup_token_embeddings(token_ids)`

源码位置：`model/hf_model.py:240-247`。

函数签名：

```python
@torch.inference_mode()
def lookup_token_embeddings(self, token_ids: List[int]) -> torch.Tensor
```

入参：

- token id 列表，不能为空。

出参：

- shape 为 `[T, H]` 的 embedding tensor。
  - `T` 是 token 数。
  - `H` 是 hidden size。

query-aware 打分直接依赖它：

- 候选 chunk token 通过它查 embedding。
- query token 也通过它查 embedding。
- 然后做余弦相似度。

需要注意：这里用的是输入 embedding，不是 Transformer 层计算后的上下文化表示。所以当前选择器是轻量的静态相似度选择器。

### 7.11 `forward_tokens(input_token_ids, past_key_values=None, position_ids=None)`

源码位置：`model/hf_model.py:249-288`。

函数签名：

```python
@torch.inference_mode()
def forward_tokens(
    self,
    input_token_ids: List[int],
    past_key_values: Any = None,
    position_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Any]
```

入参：

- `input_token_ids`：本次输入模型的 token。
- `past_key_values`：已有 KV，可为空。
- `position_ids`：显式位置 id，可为空。

出参：

- `logits`：模型输出分布。
- `past_key_values`：更新后的 KV。

行级逻辑：

- `255-256` 行：输入 token 不能为空。
- `258-260` 行：把 list 转成 shape `[1, T]` 的 tensor。
- `261` 行：读取已有 past 长度。
- `262` 行：attention 总长度等于 past 长度加本次输入长度。
- `263-265` 行：构造全 1 的 attention mask。
- `267-272` 行：如果传了 `position_ids`，检查长度并转 tensor。
- `274-281` 行：构造模型 forward 参数。
- `282-285` 行：如果模型支持，只保留最后一个 logits，降低显存占用。
- `287` 行：调用模型。
- `288` 行：返回 logits 和 KV。

在 query-aware packed 中，`position_ids` 很重要。selected token 被打包成短序列，但它们在完整 prompt 里的位置不连续，所以必须显式传入全局 position。

---

## 8. query-aware 打分

### 8.1 `_normalize_qaw_score_variant(qaw_variant)`

源码位置：`engine/inference_engine.py:81-90`。

函数签名：

```python
@staticmethod
def _normalize_qaw_score_variant(qaw_variant: str) -> str
```

当前默认行为：

- `qaw_variant="default"` 会转成 `"embedding_max"`。
- `"embedding_max"` 表示每个 chunk token 与所有 query token 分别比较，取最大相似度。

出参：

- 归一化后的变体名。

### 8.2 `_compute_query_aware_scores(...)`

源码位置：`engine/inference_engine.py:92-123`。

函数签名：

```python
def _compute_query_aware_scores(
    self,
    prompt_token_ids: List[int],
    query_token_ids: List[int],
    overlap_len: int,
    qaw_variant: str = "embedding_max",
) -> torch.Tensor
```

在当前 chunk-aware 主线中：

- `prompt_token_ids` 实际上传入的是拉平后的 `chunk_token_ids`。
- `query_token_ids` 来自 `query_text`，如果没有显式 query，则来自 `suffix_text`。
- `overlap_len` 等于 chunk token 总数。

入参：

- `prompt_token_ids`：候选 token。
- `query_token_ids`：query token。
- `overlap_len`：候选 token 数。
- `qaw_variant`：打分方式。

出参：

- shape 为 `[overlap_len]` 的分数 tensor。

行级逻辑：

- `97-98` 行：候选长度为空时返回空分数。
- `100-102` 行：query 为空时返回全 0。
- `104` 行：归一化打分变体。
- `108-110` 行：查候选 token embedding 和 query token embedding。
- `111-112` 行：对 embedding 做 L2 normalize。
- `122` 行：计算相似度矩阵。
- `123` 行：每个候选 token 取最大 query 相似度。

默认打分公式：

```text
cand_emb:  [n, h]
query_emb: [m, h]

sim = cand_emb @ query_emb.T
sim shape = [n, m]

score_i = max(sim[i, :])
```

其中：

- `n` 是 chunk token 数。
- `m` 是 query token 数。
- `h` 是 hidden size。

因为 embedding 已经 normalize，所以点积就是余弦相似度。

---

## 9. chunk 文本编码与缓存构建

### 9.1 `_encode_chunk_request(req)`

源码位置：`engine/inference_engine.py:176-194`。

函数签名：

```python
def _encode_chunk_request(
    self,
    req: GenerateRequest,
) -> Tuple[List[int], List[List[int]], List[int], List[int]]
```

入参：

- `req`：生成请求。

出参：

1. `prefix_token_ids`
2. `chunk_token_ids_list`
3. `suffix_token_ids`
4. `prompt_token_ids`

行级逻辑：

- `180-181` 行：`prefix_text` 用 `encode()` 编码，可能带 special token。
- `182-185` 行：每个 chunk 用 `encode_no_special()` 编码。
- `186-187` 行：`suffix_text` 用 `encode_no_special()` 编码。
- `188-191` 行：拼出完整 prompt token：

```text
prompt_token_ids = prefix_token_ids + 所有 chunk_token_ids + suffix_token_ids
```

当前主线里，`chunk_texts` 非空，所以通常不会走 `192-193` 行的兜底。

### 9.2 `_get_or_warm_chunk_past(req, chunk_token_ids)`

源码位置：`engine/inference_engine.py:196-218`。

函数签名：

```python
def _get_or_warm_chunk_past(
    self,
    req: GenerateRequest,
    chunk_token_ids: List[int],
) -> Tuple[Any, bool]
```

入参：

- `req`：请求对象。
- `chunk_token_ids`：某一个 chunk 的 token。

出参：

- `past_key_values`：这个 chunk 的 KV。
- `hit`：是否命中缓存。

行级逻辑：

- `199-200` 行：空 chunk 直接返回。
- `201-205` 行：构造 chunk cache key。
- `206` 行：读取 chunk 缓存。
- `207-208` 行：命中则返回缓存 KV。
- `210-211` 行：未命中时单独 prefill 该 chunk。
- `212-217` 行：把新算出的 chunk KV 写入缓存。
- `218` 行：返回 KV 和是否命中。

所谓 warm，就是第一次遇到 chunk 时先算一遍并放进缓存。后续相同 chunk 就可以直接复用。

### 9.3 `_build_chunk_reuse_context(...)`

源码位置：`engine/inference_engine.py:220-259`。

函数签名：

```python
def _build_chunk_reuse_context(
    self,
    req: GenerateRequest,
    prefix_token_ids: List[int],
    chunk_token_ids_list: List[List[int]],
) -> Tuple[Any, List[int], List[int], int, int, int]
```

入参：

- `req`：请求对象。
- `prefix_token_ids`：prefix token。
- `chunk_token_ids_list`：多个 chunk token 列表。

出参：

1. `past_key_values`：prefix + chunks 拼接后的 KV。
2. `chunk_token_ids`：所有 chunk token 拉平后的列表。
3. `chunk_global_positions`：每个 chunk token 在完整 prompt 中的位置。
4. `cache_hits`：命中的 chunk 数。
5. `cache_misses`：未命中的 chunk 数。
6. `cache_hit_tokens`：命中的 chunk token 总数。

行级逻辑：

- `225-230` 行：初始化 KV、token 列表、位置列表和统计值。
- `232-234` 行：如果 prefix 非空，先 prefill prefix。
- `236` 行：`current_len` 记录当前已经拼到完整 prompt 的长度。
- `237-239` 行：逐个处理 chunk。
- `240` 行：读取或预热该 chunk 的 KV。
- `241-245` 行：更新命中统计。
- `246-250` 行：对 chunk KV 做 RoPE 位置重定位：

```python
rebased_chunk_past = self.model_runner.rebase_past_key_values_positions(
    chunk_past,
    source_start=0,
    target_start=current_len,
)
```

- `251-252` 行：把已有 KV 和当前 chunk KV 拼接。
- `253` 行：把当前 chunk token 加入拉平列表。
- `254-255` 行：记录当前 chunk token 的全局位置。
- `256` 行：更新 `current_len`。
- `258-259` 行：返回拼接后的上下文 KV 和统计。

这一函数完成了 query-aware 前的底座构建：

```text
prefix KV + rebased chunk1 KV + rebased chunk2 KV + ...
```

---

## 10. 选择要重算的 chunk token

### 10.1 `_select_qaw_chunk_indices(req, chunk_token_ids)`

源码位置：`engine/inference_engine.py:261-289`。

函数签名：

```python
def _select_qaw_chunk_indices(
    self,
    req: GenerateRequest,
    chunk_token_ids: List[int],
) -> torch.Tensor
```

入参：

- `req`：请求对象。
- `chunk_token_ids`：所有 chunk token 拉平后的列表。

出参：

- 被选中的 chunk 内局部索引，按升序排列。

行级逻辑：

- `264` 行：读取 chunk token 总数。
- `265-266` 行：如果 chunk 为空或重算比例小于等于 0，返回空索引。
- `278` 行：选择 query 文本：

```python
query_text = req.query_text.strip() or req.suffix_text.strip()
```

也就是优先使用显式 `query_text`；没有时用 `suffix_text`。

- `279-280` 行：query 文本编码成 token。
- `281-286` 行：调用 `_compute_query_aware_scores()` 给每个 chunk token 打分。
- `288` 行：根据 `recomp_ratio` 计算要选多少 token：

```python
topk_num = min(chunk_len, max(1, int(chunk_len * req.recomp_ratio)))
```

- `289` 行：使用 `torch.topk(scores, k=topk_num)` 取分数最高的位置，并排序。

### 10.2 例子

假设 chunk token 一共 10 个，分数如下：

```text
位置:     0     1     2     3     4     5     6     7     8     9
分数:   0.1   0.8   0.2   0.7   0.05  0.4   0.9   0.3   0.6   0.15
```

如果 `recomp_ratio=0.3`：

```text
topk_num = int(10 * 0.3) = 3
```

分数最高的三个位置是：

```text
6, 1, 3
```

排序后返回：

```text
[1, 3, 6]
```

这三个位置的 chunk token 会被 packed 重算。

---

## 11. query-aware packed 重算与融合

### 11.1 `_apply_chunk_query_aware_recompute(...)`

源码位置：`engine/inference_engine.py:291-339`。

函数签名：

```python
def _apply_chunk_query_aware_recompute(
    self,
    req: GenerateRequest,
    past_key_values: Any,
    chunk_token_ids: List[int],
    chunk_global_positions: List[int],
) -> Tuple[Any, int]
```

入参：

- `req`：请求对象。
- `past_key_values`：已经拼接好的 prefix + chunks KV。
- `chunk_token_ids`：拉平后的 chunk token。
- `chunk_global_positions`：每个 chunk token 在完整 prompt 中的全局位置。

出参：

- `blended_past_key_values`：融合后的 KV。
- `recomputed_tokens`：重算 token 数。

当前主线使用 packed 分支，核心行级逻辑如下：

- `296` 行：调用 `_select_qaw_chunk_indices()` 选择 chunk 内局部位置。
- `297-298` 行：如果没有选中，直接返回原 KV。
- `312` 行：把 selected tensor 转成 Python int 列表。
- `313` 行：根据局部位置取出 selected token id：

```python
selected_token_ids = [chunk_token_ids[i] for i in selected_chunk_list]
```

- `314-316` 行：把局部位置映射成完整 prompt 全局位置：

```python
selected_global_positions = [
    chunk_global_positions[i] for i in selected_chunk_list
]
```

- `317-321` 行：把全局位置转成 tensor，后面 scatter 写回会用。
- `323-327` 行：只对 selected token 做一次 packed forward：

```python
_, selected_past_key_values = self.model_runner.forward_tokens(
    selected_token_ids,
    past_key_values=None,
    position_ids=selected_global_positions,
)
```

- `328-332` 行：把 selected KV 写回完整 KV：

```python
blended_past_key_values = scatter_selected_past_key_values(
    base_past_key_values=past_key_values,
    selected_past_key_values=selected_past_key_values,
    selected_indices=selected_global_tensor,
)
```

- `333-339` 行：记录日志并返回结果。

### 11.2 什么叫 packed

假设选中的全局位置是：

```text
[100, 560, 900]
```

它们在原 prompt 中并不连续。但 packed 做法会把这三个 token 打包成一个短输入：

```text
selected_token_ids = [token_at_100, token_at_560, token_at_900]
position_ids = [100, 560, 900]
```

然后一次 forward。

这样算出来的 selected KV 顺序是：

```text
selected KV 第 0 个 -> 写回全局位置 100
selected KV 第 1 个 -> 写回全局位置 560
selected KV 第 2 个 -> 写回全局位置 900
```

packed 的优点是计算量小。它只重算少量 token，而不是重算完整 prompt。

### 11.3 packed 的近似边界

packed 方式显式传入了全局 `position_ids`，所以位置是对齐的。但它没有完整左上下文，因为调用时：

```python
past_key_values=None
```

因此它不是 full prefill 的严格等价替代，而是一种低成本修正：

```text
底座 KV：来自 chunk full reuse
selected KV：来自少量 token 的 packed 重算
最终 KV：二者融合
```

---

## 12. suffix/query 的完整 prefill

### 12.1 `_prefill_suffix_or_last_token(...)`

源码位置：`engine/inference_engine.py:413-439`。

函数签名：

```python
def _prefill_suffix_or_last_token(
    self,
    prompt_token_ids: List[int],
    suffix_token_ids: List[int],
    past_key_values: Any,
) -> Tuple[torch.Tensor, Any]
```

入参：

- `prompt_token_ids`：完整 prompt token。
- `suffix_token_ids`：suffix/query token。
- `past_key_values`：prefix + chunks 融合后的 KV。

出参：

- `logits`：用于预测下一个 token 的分布。
- `past_key_values`：包含 suffix 后的完整 KV。

行级逻辑：

- `417` 行：读取当前 past 长度。
- `418-424` 行：如果 suffix 非空，构造 suffix 的全局位置并 forward。
- `426-431` 行：如果 prompt 只有一个 token，直接 forward。
- `433-439` 行：如果没有 suffix 且 prompt 多于一个 token，则保留前 `len(prompt)-1` 的 KV，补算最后一个 token 的 logits。

当前 query-aware 主线中，suffix 通常包含 query 或回答格式要求，所以它会在 QAW 融合后的 KV 后面完整 prefill。

---

## 13. chunk-aware 总流程：`_generate_chunk_aware(req)`

源码位置：`engine/inference_engine.py:441-513`。

函数签名：

```python
def _generate_chunk_aware(self, req: GenerateRequest) -> GenerateResult
```

入参：

- `req`：chunk-aware 请求。

出参：

- `GenerateResult`。

### 13.1 初始化

源码位置：`engine/inference_engine.py:443-453`。

流程：

1. `443` 行：记录开始时间。
2. `444` 行：初始化首 token 延迟。
3. `445` 行：解析重算策略。
4. `447-448` 行：编码请求，得到 prefix/chunk/suffix/prompt token。
5. `450-453` 行：初始化 chunk 命中和重算统计。

### 13.2 不使用缓存时

源码位置：`engine/inference_engine.py:455-458`。

```python
if not req.use_cache:
    logits, past_key_values = self.model_runner.forward_tokens(
        prompt_token_ids, past_key_values=None)
    recompute_mode = "full_prefill"
```

这条路径直接完整 prefill，不使用 query-aware。

### 13.3 使用缓存并启用 query-aware

源码位置：`engine/inference_engine.py:459-481`。

主流程：

1. `460-466` 行：调用 `_build_chunk_reuse_context()`，构建 prefix + chunks KV。
2. `467` 行：先把模式记为 `chunk_full_reuse`。
3. `468-476` 行：如果 `recompute_strategy == "query_aware"`，调用 `_apply_chunk_query_aware_recompute()` 做 packed 重算和融合。
4. `477-481` 行：调用 `_prefill_suffix_or_last_token()` 计算 suffix/query 或补 logits。

### 13.4 decode

源码位置：`engine/inference_engine.py:483-495`。

流程：

1. 初始化 `generated_ids`。
2. 获取 EOS token id。
3. 循环最多 `max_new_tokens` 次。
4. 用 `_sample_next_token()` 从 logits 中选下一个 token。
5. 记录第一个 token 的耗时。
6. 如果遇到 EOS，则提前停止。
7. 把新 token 输入模型，更新 logits 和 KV。

### 13.5 返回结果

源码位置：`engine/inference_engine.py:497-513`。

返回内容包括：

- 生成文本。
- 完整文本。
- 命中的 chunk token 数。
- prompt token 数。
- 生成 token 数。
- 总耗时。
- 首 token 耗时。
- 当前执行模式。
- query-aware packed 重算 token 数。
- chunk 数、命中数、未命中数。

---

## 14. 采样函数：`_sample_next_token`

源码位置：`engine/inference_engine.py:39-67`。

函数签名：

```python
def _sample_next_token(
    self,
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int
```

入参：

- `logits`：模型输出的 token 分布。
- `temperature`：采样温度。
- `top_p`：累计概率截断阈值。

出参：

- 下一个 token id。

当前默认 `temperature=0.0`，所以执行贪心：

```python
return int(torch.argmax(last_logits, dim=-1).item())
```

如果温度大于 0，则会：

1. logits 除以 temperature。
2. softmax 得到概率。
3. 如果 `top_p < 1.0`，保留累计概率范围内的候选 token。
4. 用 `torch.multinomial()` 采样。

---

## 15. 完整 packed QAW 伪代码

下面用伪代码把当前主线串起来：

```python
def query_aware_packed_generate(req):
    # 1. 编码请求
    prefix_ids, chunk_ids_list, suffix_ids, prompt_ids = encode_chunk_request(req)

    # 2. prefix 正常 prefill
    past = forward(prefix_ids) if prefix_ids else None

    flat_chunk_ids = []
    flat_chunk_positions = []
    current_len = len(prefix_ids)

    # 3. 逐个 chunk 读取或预热 KV
    for chunk_ids in chunk_ids_list:
        key = build_chunk_key(req.chunk_namespace, model_name, chunk_ids)
        chunk_past = chunk_cache.get(key)

        if chunk_past is None:
            chunk_past = forward(chunk_ids)
            chunk_cache.put(key, chunk_past)

        # 4. chunk KV 从本地位置移动到全局位置
        chunk_past = rebase_past_key_values_positions(
            chunk_past,
            source_start=0,
            target_start=current_len,
        )

        # 5. 拼接 KV
        past = concat_past_key_values(past, chunk_past)

        flat_chunk_ids.extend(chunk_ids)
        flat_chunk_positions.extend(range(current_len, current_len + len(chunk_ids)))
        current_len += len(chunk_ids)

    # 6. query-aware 打分
    query_text = req.query_text or req.suffix_text
    query_ids = encode_no_special(query_text)
    scores = compute_query_aware_scores(flat_chunk_ids, query_ids)

    # 7. 取 Top-K
    k = max(1, int(len(flat_chunk_ids) * req.recomp_ratio))
    selected_local_indices = topk(scores, k)

    # 8. 局部位置映射到全局位置
    selected_ids = [flat_chunk_ids[i] for i in selected_local_indices]
    selected_positions = [flat_chunk_positions[i] for i in selected_local_indices]

    # 9. packed 重算
    selected_past = forward(
        selected_ids,
        past_key_values=None,
        position_ids=selected_positions,
    )

    # 10. scatter 写回完整 KV
    past = scatter_selected_past_key_values(
        base_past_key_values=past,
        selected_past_key_values=selected_past,
        selected_indices=selected_positions,
    )

    # 11. suffix/query 完整 prefill
    logits, past = forward(
        suffix_ids,
        past_key_values=past,
        position_ids=range(current_len, current_len + len(suffix_ids)),
    )

    # 12. decode
    generated_ids = []
    for step in range(req.max_new_tokens):
        next_id = sample_next_token(logits)
        generated_ids.append(next_id)
        logits, past = forward([next_id], past_key_values=past)

    return decode(generated_ids)
```

---

## 16. 当前主线函数速查表

### 16.1 请求与结果

| 名称 | 类型 | 当前作用 |
|---|---|---|
| `GenerateRequest` | dataclass | 描述一次 chunk-aware query-aware 请求 |
| `GenerateResult` | dataclass | 描述生成文本和统计信息 |

### 16.2 chunk 缓存

| 名称 | 入参 | 出参 | 当前作用 |
|---|---|---|---|
| `ChunkKV` | dataclass 字段 | 对象 | 保存单个 chunk 的 token 和 KV |
| `build_chunk_key` | `namespace`, `model_name`, `token_ids` | `str` | 构造 chunk 缓存 key |
| `_evict_expired_chunks` | 无 | `None` | 删除过期 chunk |
| `get_chunk` | `cache_key` | `Optional[ChunkKV]` | 读取 chunk KV |
| `put_chunk` | `cache_key`, `namespace`, `token_ids`, `past_key_values` | `None` | 写入 chunk KV |
| `clear_chunks` | `namespace` 可选 | `None` | 清理 chunk KV |
| `stats` | 无 | `Dict[str, int]` | 返回缓存统计 |

### 16.3 KV 融合

| 名称 | 入参 | 出参 | 当前作用 |
|---|---|---|---|
| `concat_past_key_values` | `left`, `right` | KV | 拼接 prefix 与 chunk KV |
| `clone_past_key_values` | `past_key_values` | KV | 克隆 KV，避免污染缓存 |
| `scatter_selected_past_key_values` | `base`, `selected`, `selected_indices` | KV | 把重算 KV 写回完整 KV |

### 16.4 模型封装

| 名称 | 入参 | 出参 | 当前作用 |
|---|---|---|---|
| `_resolve_dtype` | `dtype_name` | `torch.dtype` | 解析模型精度 |
| `HFModelRunner.__init__` | 模型名、设备、精度、日志器等 | `None` | 加载 tokenizer 和模型 |
| `encode` | `text` | `List[int]` | 编码 prefix |
| `encode_no_special` | `text` | `List[int]` | 编码 chunk、suffix、query |
| `decode` | `token_ids` | `str` | 解码生成结果 |
| `eos_token_id` | 无 | `int` | 判断生成停止 |
| `_find_rotary_inv_freq` | 无 | `torch.Tensor` | 获取 RoPE 参数 |
| `_rotate_half` | `x` | `torch.Tensor` | RoPE 辅助操作 |
| `_apply_rope_to_key` | `key`, `positions` | `torch.Tensor` | 应用目标位置 RoPE |
| `_unapply_rope_from_key` | `key`, `positions` | `torch.Tensor` | 移除源位置 RoPE |
| `rebase_past_key_values_positions` | `past_key_values`, `source_start`, `target_start` | KV | chunk KV 位置重定位 |
| `get_past_len` | `past_key_values` | `int` | 读取 KV 长度 |
| `truncate_past_key_values` | `past_key_values`, `target_len` | KV | 无 suffix 时补 logits |
| `lookup_token_embeddings` | `token_ids` | `torch.Tensor` | query-aware 打分 |
| `forward_tokens` | `input_token_ids`, `past_key_values`, `position_ids` | `(logits, KV)` | 执行模型前向 |

### 16.5 推理引擎

| 名称 | 入参 | 出参 | 当前作用 |
|---|---|---|---|
| `InferenceEngine.__init__` | `model_runner`, `kv_cache_manager`, `logger` | `None` | 注入依赖 |
| `_sample_next_token` | `logits`, `temperature`, `top_p` | `int` | 选择下一个 token |
| `_normalize_qaw_score_variant` | `qaw_variant` | `str` | 归一化打分方式 |
| `_compute_query_aware_scores` | 候选 token、query token、长度、变体 | `torch.Tensor` | 给 chunk token 打分 |
| `_encode_chunk_request` | `req` | 四类 token | 编码 chunk-aware 请求 |
| `_get_or_warm_chunk_past` | `req`, `chunk_token_ids` | `(KV, bool)` | 读取或预热 chunk KV |
| `_build_chunk_reuse_context` | `req`, `prefix_token_ids`, `chunk_token_ids_list` | KV、token、位置、统计 | 构建 prefix + chunks KV |
| `_select_qaw_chunk_indices` | `req`, `chunk_token_ids` | `torch.Tensor` | 选 Top-K chunk token |
| `_apply_chunk_query_aware_recompute` | `req`, KV、chunk token、全局位置 | `(KV, int)` | packed 重算并写回 |
| `_prefill_suffix_or_last_token` | prompt token、suffix token、KV | `(logits, KV)` | 计算 suffix/query |
| `_generate_chunk_aware` | `req` | `GenerateResult` | 当前主流程 |
| `generate` | `req` | `GenerateResult` | 总入口，分发到 chunk-aware |

---

## 17. 当前实现的近似边界

### 17.1 chunk 独立 KV 的近似

chunk 单独 prefill 时，没有完整左上下文。RoPE 重定位只修正位置信息，不会补回缺失的上下文。因此：

```text
重定位后的 chunk KV 不严格等于 full prefill 中同一位置的 KV。
```

### 17.2 packed 重算的近似

packed 重算只输入 selected token，并传入它们的全局位置。它位置正确，但没有完整左上下文。因此：

```text
packed selected KV 也不严格等于 full prefill 中这些 token 的 KV。
```

它是一种成本很低的修正方式。

### 17.3 embedding 打分的近似

当前 query-aware 使用输入 embedding 相似度选 token。它便宜，但不是完整语义理解。

它更擅长捕捉：

- query 中出现的关键词。
- 与 query token embedding 接近的 chunk token。

它不擅长：

- 多跳推理。
- 复杂指代。
- 否定语义。
- 上下文中动态变化的词义。

---

## 18. 初学者最容易混淆的点

### 18.1 token id 和 token 位置不同

`token_id` 是词表编号，例如 `15123`。

token 位置是它在当前 prompt 里的序号，例如第 100 个 token。

query-aware 先选择位置，再根据位置找到 token id 重算。

### 18.2 chunk 局部位置和 prompt 全局位置不同

`_select_qaw_chunk_indices()` 返回的是 chunk 拉平后的局部位置。

`_apply_chunk_query_aware_recompute()` 会把它映射到完整 prompt 的全局位置。

packed forward 必须使用全局位置作为 `position_ids`。

### 18.3 query_text 优先于 suffix_text

当前选择 query 的代码是：

```python
query_text = req.query_text.strip() or req.suffix_text.strip()
```

因此：

- 显式传了 `query_text`，就用它。
- 没传 `query_text`，才用 `suffix_text`。

### 18.4 suffix 不参与 chunk Top-K 选择

Top-K 的候选范围是 chunk token。suffix 会在融合后的 KV 后面完整 prefill。

这符合当前主线的意图：chunk 是被缓存复用的内容，suffix/query 是当前请求的新增条件。

---

## 19. 一句话总结

当前项目实际使用的 query-aware packed 可以概括为：

```text
先复用并拼接 chunk KV；
再用 query embedding 找出最相关的 chunk token；
只重算这些 token；
把新 KV 写回原位置；
最后完整计算 suffix/query 并进入生成。
```

最核心的函数是：

- `InferenceEngine._generate_chunk_aware()`
- `InferenceEngine._build_chunk_reuse_context()`
- `InferenceEngine._compute_query_aware_scores()`
- `InferenceEngine._select_qaw_chunk_indices()`
- `InferenceEngine._apply_chunk_query_aware_recompute()`
- `scatter_selected_past_key_values()`
- `HFModelRunner.rebase_past_key_values_positions()`
- `HFModelRunner.forward_tokens()`
