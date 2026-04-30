# CacheBlend 代码级原理说明

本文档按“数据结构 -> 缓存层 -> 模型层 -> 推理层 -> 实验脚本”的顺序，整理当前仓库中 CacheBlend 核心实现的工作原理。目标不是重复 README 的运行说明，而是把以下文件背后的算法、关键代码、缓存语义、复杂度和近似假设讲清楚：

- `schema/types.py`
- `utils/logging_utils.py`
- `cache/kv_cache.py`
- `cache/kv_fusion.py`
- `model/hf_model.py`
- `engine/inference_engine.py`
- 最后用 `example/blend_musique.py` 串起完整测试流程

## 1. 涉及文件与代码规模

统计口径使用 `wc -l`。

### 1.1 核心实现文件

| 文件 | 行数 | 作用 |
|---|---:|---|
| `cache/kv_cache.py` | 178 | session 级 KV 缓存与 chunk 级 KV 缓存管理 |
| `cache/kv_fusion.py` | 66 | KV 的拼接、切片、克隆、散射写回 |
| `engine/inference_engine.py` | 885 | 主算法：full prefill / full reuse / KV-diff / QAW |
| `model/hf_model.py` | 238 | HF 模型封装、RoPE 重定位、token embedding 查询 |
| `schema/types.py` | 61 | 请求与结果数据结构 |
| `utils/logging_utils.py` | 29 | 统一日志格式 |

**上述 6 个核心文件合计 1457 行。**

### 1.2 实验脚本示例

| 文件 | 行数 | 作用 |
|---|---:|---|
| `example/blend_musique.py` | 392 | MusiQue 数据集对比实验脚本 |

**若把 `blend_musique.py` 一并计入，总计 1849 行。**

## 2. 系统总览

从功能上看，当前仓库可以分成 4 层：

1. **请求/结果层**  
   `GenerateRequest` 和 `GenerateResult` 定义算法输入输出。
2. **缓存层**  
   `KVCacheManager` 管理两类缓存：
   - `session` 级完整前缀缓存
   - `chunk` 级独立文本块缓存
3. **模型层**  
   `HFModelRunner` 负责 tokenizer、前向、embedding 查询、RoPE 位置重定位。
4. **推理控制层**  
   `InferenceEngine` 负责：
   - full prefill
   - exact prefix reuse
   - chunk full reuse
   - KV-diff 重算
   - query-aware 重算
   - decode 与缓存写回

最重要的一点是：当前仓库里其实同时存在两类“复用”语义。

- **session 级复用**：复用过去某次完整 prompt 的 `past_key_values`。这是接近标准 HuggingFace KV reuse 的用法，语义最稳。
- **chunk 级复用**：把每个 chunk 单独算出的 KV 拿出来，经过 RoPE 位置重定位后再拼回长上下文。这个路径是 CacheBlend 的核心实验对象，但它不是严格等价于完整 prefill，而是带近似假设的实现。

后文会反复看到这条边界：**session 路径偏精确，chunk 路径偏实验性近似。**

## 3. 数据结构层：`types.py`

`schema/types.py` 很短，但它决定了整个引擎的参数面。

### 3.1 `GenerateRequest`

可以把它拆成 4 组字段理解：

#### A. 基础生成参数

- `session_id`
- `prompt`
- `max_new_tokens`
- `temperature`
- `top_p`
- `use_cache`

这一组控制“普通的因果语言模型生成”。

#### B. 重算策略参数

- `recompute_strategy`: `none / kv_diff / query_aware`
- `enable_kv_diff_recompute`: 兼容旧字段
- `recomp_ratio`
- `suffix_len`
- `query_text`
- `qaw_variant`
- `qaw_type`
- `qaw_random_seed`

这一组控制“遇到旧缓存但不能直接复用时，如何做选择性重算”。

#### C. chunk-aware 参数

- `prefix_text`
- `chunk_texts`
- `suffix_text`
- `chunk_namespace`

只要 `chunk_texts is not None`，`InferenceEngine.generate()` 就会切换到 chunk-aware 主线。也就是说，是否进入 CacheBlend 的 chunk 实验路径，不是由另一个显式开关控制，而是由请求结构本身决定。

### 3.2 `GenerateResult`

结果结构除了文本输出外，最重要的是统计信息：

- `reused_prefix_tokens`
- `recompute_mode`
- `recomputed_tokens`
- `chunk_count`
- `chunk_cache_hits`
- `chunk_cache_misses`
- `first_token_latency_s`
- `total_latency_s`

这些字段就是实验脚本最终拿来做 TTFT、总时延、重算比例和命中率分析的数据基础。

## 4. 日志层：`logging_utils.py`

`utils/logging_utils.py` 逻辑很简单，但它解决了一个常见工程问题：重复创建 logger 时避免重复绑 handler，导致日志刷屏。

关键点：

- 只在 `not logger.handlers` 时绑定 `StreamHandler`
- 默认格式统一为：
  - 时间
  - 级别
  - logger 名
  - 文本
- 默认日志级别读取 `LOG_LEVEL`
- `logger.propagate = False`，避免冒泡到 root logger

这个文件不涉及算法，但对实验排障有帮助，尤其是你想看某个 session 是否命中缓存、是否退化到 full prefill、是否触发 OOM fallback 时。

## 5. 缓存层：`kv_cache.py`

`cache/kv_cache.py` 是整个仓库最基础的状态持有层。

### 5.1 两类缓存单元

定义了两个 dataclass：

```python
@dataclass
class SessionKV:
    token_ids: List[int]
    past_key_values: Any
    next_token_logits: Any
    last_access_ts: float

@dataclass
class ChunkKV:
    cache_key: str
    namespace: str
    token_ids: List[int]
    past_key_values: Any
    last_access_ts: float
```

它们分别对应：

- **SessionKV**：一整段 prompt 甚至 `prompt + generated tokens` 的缓存
- **ChunkKV**：某个独立文本块单独预填充后的缓存

### 5.2 为什么需要 `next_token_logits`

`SessionKV` 比 `ChunkKV` 多了一个 `next_token_logits`。这个字段的作用是：

- 当某个 prompt 已经完整 warm 过，且 `max_new_tokens=0`
- 下一次完全相同的 prompt 再来时
- 不必重新补算“最后一个 prompt token 的 logits”

这是一个小优化，但很实用，因为完整 prompt 已经有缓存时，进入 decode 只需要下一 token 分布即可。

### 5.3 缓存淘汰策略

当前实现采用：

- `TTL`
- `LRU`
- 基于 `OrderedDict`

这意味着它是一个**单进程、内存内、无持久化**的缓存器。它没有：

- 多进程一致性
- 跨机器共享
- 页式管理
- 显存/内存分层调度

所以它更像一个实验缓存层，而不是生产级缓存服务。

### 5.4 `build_chunk_key()` 的语义

```python
payload = ",".join(str(t) for t in token_ids)
digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
model_digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
return f"{namespace}:{model_digest}:{digest}"
```

chunk key 由三部分组成：

1. `namespace`
2. `model_name` 哈希
3. `token_ids` 哈希

这样设计有两个直接好处：

- 不同实验命名空间不会串数据
- 同一个文本块在不同模型下不会误复用

### 5.5 设备语义：缓存对象在 Python 内存里，但 tensor 通常在 GPU 上

这个点非常关键。

`KVCacheManager` 自身只是 Python 的 `OrderedDict`。但它保存的 `past_key_values` 并不会自动 `.cpu()`。也就是说：

- dict 结构在主机进程内存中
- 但其中的 tensor 仍然停留在原设备上
- 默认 `DEVICE=cuda` 时，KV 实际占用的是 GPU 显存

因此，`kv_cache.py` 虽然类名和注释写的是“内存 KV 缓存管理器”，但更准确地说是：

> 它是一个进程内对象缓存管理器；缓存的张量驻留在哪个 device，取决于模型前向时生成它们的 device。

### 5.6 复杂度

设 session 数为 `S`，chunk 数为 `C`。

- `get()/put()/get_chunk()/put_chunk()` 的字典访问平均是 `O(1)`
- 但淘汰时会扫描过期项：
  - `session` 过期扫描 `O(S)`
  - `chunk` 过期扫描 `O(C)`

在当前实验规模下这不是瓶颈；瓶颈几乎总在模型前向和 KV tensor 拷贝上。

## 6. KV 操作层：`kv_fusion.py`

`cache/kv_fusion.py` 是一些纯张量级别的 KV 变换函数。它们的共同特点是：**逻辑简单，但每次操作都伴随大 tensor 拷贝。**

### 6.1 `concat_past_key_values`

作用：沿序列维把两段 KV 拼起来。

```python
merged_k = torch.cat([left_k, right_k], dim=-2)
merged_v = torch.cat([left_v, right_v], dim=-2)
```

这是 chunk full reuse 能把 prefix 和多个 chunk 接成一整段上下文的基础。

### 6.2 `clone_past_key_values`

作用：避免调用方在修改 KV 时污染缓存池里的原对象。

这是必要的，因为 chunk 缓存会被重复读取，如果不 clone，后续的 scatter 或裁剪可能破坏“原始 chunk checkpoint”。

### 6.3 `slice_past_key_values`

作用：取 `[start_idx:end_idx)` 区间的 KV。

主要用于 window QAW：先对某个窗口连同左上下文一起重算，再从完整重算结果里切出窗口对应的那一段 KV。

### 6.4 `scatter_selected_past_key_values`

作用：把“打包重算得到的 selected KV”写回 base KV 的对应位置。

```python
base_k[:, :, selected_indices, :] = selected_k[:, :, :selected_num, :]
base_v[:, :, selected_indices, :] = selected_v[:, :, :selected_num, :]
```

这个函数是 QAW packed 路径的核心。思想上相当于：

1. 先只算选中的 token
2. 再把这些 token 的 K/V 填回原长序列位置
3. 未选中的 token 继续复用旧 KV

### 6.5 复杂度

假设单层 KV 张量形状约为 `[H, L, D]`。

- `concat`: `O(L_left + L_right)` 大小的数据拷贝
- `clone`: `O(L)`
- `slice`: `O(slice_len)`
- `scatter`: `O(selected_len)` 写入，但由于先 `clone base`，总体常常接近 `O(L)`

结论很直接：

> `kv_fusion.py` 的瓶颈不是 Python 控制流，而是显存带宽和 tensor 拷贝。

这也是为什么当前实现虽然逻辑直观，但并不轻量。

## 7. 模型层：`hf_model.py`

`model/hf_model.py` 把 HuggingFace Causal LM 包成一个统一接口，关键职责有四个：

1. 加载模型和 tokenizer
2. 提供 token 编解码
3. 提供标准 `forward_tokens()`
4. 提供 RoPE 位置重定位和 embedding 查询

## 7.1 模型加载策略

初始化阶段做了几件事：

- 如果请求 `cuda` 但当前不可用，回退到 `cpu`
- 在 CPU 上强制把 `bf16/fp16` 改成 `fp32`
- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForCausalLM.from_pretrained(...)`
- `self.model.to(self.device)`
- `self.model.eval()`

这说明当前实现默认是“单卡单模型、推理态、不考虑训练和梯度”。

## 7.2 为什么 `encode()` 和 `encode_no_special()` 要分开

```python
def encode(self, text: str) -> List[int]:
    return self.tokenizer.encode(text, add_special_tokens=True)

def encode_no_special(self, text: str) -> List[int]:
    return self.tokenizer.encode(text, add_special_tokens=False)
```

这对 chunk 路径非常重要。

- `prefix_text` 通常会带 BOS 等特殊符号
- `chunk_texts` 和 `suffix_text` 不应该每段都重复插入特殊符号

所以 `_encode_chunk_request()` 里采用的是：

- `prefix`: `encode`
- `chunks`: `encode_no_special`
- `suffix`: `encode_no_special`

否则 chunk 间会莫名插入 BOS/EOS，导致 token 序列根本不等价。

## 7.3 `forward_tokens()`：项目里真正的前向入口

这是整个仓库最重要的模型接口。

```python
outputs = self.model(**model_kwargs)
return outputs.logits, outputs.past_key_values
```

它统一封装了：

- `input_ids`
- `attention_mask`
- `position_ids`
- `past_key_values`
- `use_cache=True`

因此，项目中无论是 full prefill、full reuse、QAW packed、window 重算，最后都要落回到这个接口。

### 7.3.1 为什么 decode 一定会用到 KV cache

只要 `past_key_values` 非空，下一步前向时：

- 模型不再重新算整段历史 token 的 K/V
- 而是只处理新输入 token
- 并把新 token 的 K/V 追加到旧缓存之后

这正是标准自回归 KV cache 的定义。

### 7.3.2 `num_logits_to_keep`

```python
if self._supports_num_logits_to_keep:
    model_kwargs["num_logits_to_keep"] = 1
```

这一步是一个很实际的显存优化：

- 长 prompt full prefill 时，完整 logits 形状本来可能是 `[1, seq_len, vocab]`
- 但生成阶段通常只需要最后一个位置的 logits

如果模型支持该参数，就只保留最后一步，能显著降低显存峰值。

## 7.4 QAW 用到的 embedding 是什么

`lookup_token_embeddings()` 的实现是：

```python
ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
emb_layer = self.model.get_input_embeddings()
return emb_layer(ids)
```

因此当前 QAW 打分使用的是：

- **模型输入 embedding**
- 不是最后一层 hidden state
- 不是独立 embedding 模型
- 不是上下文化表示

这意味着当前 QAW 更像一个**静态词嵌入相似度选择器**，而不是语义检索器。

## 7.5 RoPE 重定位：chunk reuse 的关键理论

### 7.5.1 问题背景

如果一个 chunk 是独立预填充得到的，那么它在计算时的位置通常是：

- 第一个 token 在 position 0
- 第二个 token 在 position 1
- ...

但真正把它拼到完整 prompt 中时，它的全局位置往往不是从 0 开始，而是：

- 第一个 token 在 `offset`
- 第二个 token 在 `offset + 1`
- ...

对于使用 RoPE 的模型，**Key 的方向和位置绑定**。所以直接拿“局部位置下得到的 K”去当“全局位置下的 K”会不一致。

### 7.5.2 当前实现的做法

`rebase_past_key_values_positions()` 的核心逻辑是：

1. 对原始 key 做反旋转，得到“去掉 source position 后的 key”
2. 再按 target position 重新旋转
3. value 直接 clone，不做位置重排

形式上就是：

`K_target = Rope(target_pos, Rope^{-1}(source_pos, K_source))`

### 7.5.3 这解决了什么，没有解决什么

它解决了：

- **位置编码不一致问题**

它没有解决：

- **chunk 独立计算时缺失左上下文的问题**

也就是说，RoPE rebase 只能把“位置信息”校正回来，但不能让一个原本孤立计算的 chunk，直接变成“在完整前缀条件下算出来的 chunk”。

这也是为什么 chunk full reuse 是近似，不是严格等价。

### 7.5.4 复杂度

设 chunk 长度为 `L`，层数为 `N`，每层 key 大小按 `O(L * D)` 估计。

- `rebase_past_key_values_positions()` 大致是 `O(N * L * D)`

这里没有 attention 计算，但存在大规模张量变换，因此也不便宜。

## 8. 推理层：`inference_engine.py`

`engine/inference_engine.py` 是整个仓库的核心。

从功能上看，它包含三条主要路线：

1. **legacy session 路线**
   - full prefill
   - exact prefix reuse
   - KV-diff
   - legacy QAW
2. **chunk-aware 路线**
   - chunk full reuse
   - chunk QAW packed/window
3. **统一 decode**

下面按逻辑拆解。

## 8.1 采样逻辑

`_sample_next_token()` 很直接：

- `temperature <= 0` 时 greedy
- 否则做 top-p 采样

这部分不是本文重点，但它提醒一个事实：

> 当前实验的性能差异主要来自 prefill / reuse / recompute，不来自采样策略。

## 8.2 评分函数

### 8.2.1 KV-diff 分数

```python
layer_score = (new_v - old_v).pow(2).sum(dim=(0, 1, 3))
```

这里对每一层的 `V` 做：

- 逐 token
- 跨 batch、heads、head_dim
- L2 差异累加

得到长度为 `overlap_len` 的 token 分数。

理论含义是：

> 如果某个 token 在旧 prompt 和新 prompt 下产生的 Value 差异很大，那么它可能是“受上下文变化影响较大”的位置，更值得重算。

#### 复杂度

若重叠长度为 `L_o`，层数 `N`，KV 头数 `H_kv`，head_dim 为 `D_h`：

`O(N * H_kv * L_o * D_h)`

这本身不轻，但它仍然比完整模型前向便宜；真正的问题在于它之前先做了一次完整新 prompt prefill。

### 8.2.2 Query-aware 分数

```python
cand_emb = lookup(prompt_token_ids[:overlap_len])
query_emb = lookup(query_token_ids)
cand_emb = normalize(cand_emb)
query_emb = normalize(query_emb)
sim = cand_emb @ query_emb.transpose(0, 1)
score = sim.max(dim=1).values
```

它的语义是：

> 对每个候选 token，找它和 query 中最相似的那个 query token，取最大余弦相似度作为分数。

这是一种典型的 `MaxSim` 风格 late interaction。

#### 理论特点

- 保留 query 中局部关键词的激活能力
- 不会像均值池化那样把实体词稀释掉
- 但使用的是静态 input embedding，不是 contextual embedding

#### 复杂度

若 chunk token 数为 `n`，query token 数为 `m`，hidden size 为 `d`：

- 时间复杂度近似 `O(n * m * d)`
- 相似度矩阵额外空间 `O(n * m)`

这不是免费的，但在实验里 `m` 通常较小，且通常仍比一次完整模型重算便宜。

## 8.3 通用 token 选择器

`_select_token_indices_from_scores()` 是很多分支共用的选择器。

它最终选中的位置集合由几部分并起来：

- `topk`: 评分前 K 的位置
- `forced`: 尾部强制重算区
- `added`: 新增 token
- `changed`: 与旧 prompt 不同的位置，可选

这里隐含了一个工程判断：

> 即使分数低，尾部 token 和新增 token 也往往最危险，因此不能完全交给排序分数决定。

这和很多实际系统的“启发式兜底”一致。

## 8.4 chunk-aware 编码与缓存预热

### 8.4.1 `_encode_chunk_request()`

它把请求拆成：

- `prefix_token_ids`
- `chunk_token_ids_list`
- `suffix_token_ids`
- `prompt_token_ids`

这一步定义了 chunk-aware 路线的 token 边界。

### 8.4.2 `_get_or_warm_chunk_past()`

逻辑非常清晰：

1. 用 token hash 生成 chunk key
2. 命中则直接返回 `entry.past_key_values`
3. miss 则对 chunk 独立做一次 `forward_tokens(chunk, None)`
4. 把得到的 `past_key_values` 写入 chunk cache

#### 理论含义

它缓存的是：

> “该 chunk 作为独立序列时”的 KV

而不是：

> “该 chunk 处在某个特定全局前缀下时”的 KV

这正是 chunk reuse 后续需要 RoPE rebase、仍然只能近似的根源。

## 8.5 构造 chunk full reuse 上下文

`_build_chunk_reuse_context()` 逻辑是：

1. 先对 `prefix_text` 做一次前向，得到 prefix KV
2. 逐 chunk：
   - 获取独立 chunk KV
   - 做 RoPE 重定位
   - 拼接到当前 `past_key_values`
3. 记录命中次数、miss 次数和命中 token 数

### 8.5.1 理论解释

这里试图构造的是：

`KV(prefix + chunk_1 + chunk_2 + ... + chunk_n)`

但实际构造方式不是一次完整前向，而是：

`KV(prefix)`  
`+ rebase(KV(chunk_1_alone))`  
`+ rebase(KV(chunk_2_alone))`  
`+ ...`

因此它不是严格的“全量前缀真实 KV”，而是“分块独立 KV 经位置对齐后的拼装体”。

### 8.5.2 复杂度与当前实现的代价

设有 `C` 个 chunk，每个 chunk 长度为 `l_i`，总长度 `L = sum l_i`。

成本包括：

1. **chunk miss 的独立 prefill**
   - `sum O(l_i^2)` 级 attention 成本
2. **RoPE rebase**
   - `sum O(l_i)` 量级 tensor 变换
3. **反复 `torch.cat` 拼接**
   - 当前实现每拼一次都会复制已有大块 tensor
   - 因此拼接开销接近 `O(l_1 + (l_1+l_2) + ... + L)`
   - 最坏情况下是 `O(C * L)` 级别的数据搬运，均匀 chunk 时接近平方级

也就是说，当前 chunk full reuse 的瓶颈不仅在模型前向，也在**重复张量复制**。

## 8.6 chunk Query-Aware 重算

当前 chunk 路径的核心实验分为两种：

- `packed`
- `window`

### 8.6.1 `packed` 方案

流程如下：

1. 用 QAW 分数选出若干 chunk token 索引
2. 取出这些 token 的 `token_ids`
3. 用它们的全局位置作为 `position_ids`
4. 做一次打包前向：

```python
forward_tokens(selected_token_ids, past_key_values=None, position_ids=selected_global_positions)
```

5. 用 `scatter_selected_past_key_values()` 把重算得到的 KV 写回原位置

#### 理论优点

- 只重算 `K` 个 token
- 若 `K << L`，前向长度显著下降

#### 理论缺陷

这里虽然给了正确的全局 `position_ids`，但它并没有给 selected token 提供完整左上下文。  
因此被重算的 token 在前向时只能看到：

- 选中的其他 token
- 但看不到原始未选中的左邻居 token

所以它只是：

> 位置对齐的稀疏重算

而不是：

> 在真实完整上下文下的局部精确重算

这会带来质量偏差。

#### 复杂度

设选中的 token 数为 `K`：

- QAW 打分：`O(n * m * d)`
- packed 前向：attention 近似 `O(K^2)`
- scatter 写回：约 `O(L)` 级 tensor 复制/写入

### 8.6.2 `window` 方案

为缓解 packed 的“缺左上下文”问题，仓库又加了 `window` 方案：

1. 先选中若干关键 token
2. 每个关键 token 向左扩 16 个 token
3. 合并重叠窗口
4. 对每个窗口：
   - 取 `prefix_past = truncate(blended_past, global_start)`
   - 在真实左上下文下连续重算窗口 token
   - 再切出窗口范围 KV 写回

#### 理论含义

window 方案实际上在做：

> “只对少数危险位置所在的小连续片段，在真实左上下文下做局部精确重算”

相比 packed，它更接近真实前向。

#### 复杂度

若窗口集合为 `W_1, W_2, ..., W_r`，第 `i` 个窗口长度为 `w_i`，左上下文长度为 `p_i`：

- 第 `i` 个窗口前向大致成本是 `O(p_i * w_i + w_i^2)`
- 总成本为 `sum O(p_i * w_i + w_i^2)`

通常它比 packed 更贵，但质量更可靠。

## 8.7 `_prefill_suffix_or_last_token()`

chunk KV 拼完之后，还不能直接 decode，因为还需要拿到“进入 decode 的最后一步 logits”。

当前实现有两种情况：

1. **有 suffix**  
   直接把 `suffix_token_ids` 在已拼好的 `past_key_values` 后面跑一遍
2. **没有 suffix**  
   只补算最后一个 prompt token 的 logits

这一步的工程作用是：

> 把 chunk 复用上下文真正接回标准自回归 decode 起点。

## 8.8 `_generate_chunk_aware()`：chunk 主线

这条主线可以概括为：

### A. `use_cache=False`

- 不走 chunk cache
- 直接对完整 prompt full prefill
- `recompute_mode = "full_prefill"`

### B. `use_cache=True` + `recompute_strategy="none"`

- 走 chunk KV 读取 / 预热
- 构建拼接上下文
- 不做 QAW
- `recompute_mode = "chunk_full_reuse"`

### C. `use_cache=True` + `recompute_strategy="query_aware"`

- 先构建 chunk full reuse 上下文
- 再对 chunk token 做 QAW 选择性重算
- `recompute_mode = "chunk_query_aware_recompute"`

无论哪个分支，最后都会进入统一 decode：

```python
logits, past_key_values = self.model_runner.forward_tokens(
    [next_token_id], past_key_values=past_key_values)
```

## 8.9 legacy session 路线：`generate()`

如果 `req.chunk_texts is None`，引擎走的是另一条路线。

### 8.9.1 exact prefix reuse

命中条件：

```python
cached_entry is not None and self._is_prefix(cached_entry.token_ids, prompt_token_ids)
```

一旦命中：

- 直接复用 `cached_entry.past_key_values`
- 只对剩余增量 token 做 prefill
- 若 prompt 完全一致且带 `next_token_logits`，甚至可以 0 prompt 计算直接进入 decode

这一条是当前仓库里最接近标准 KV cache 的精确复用模式。

### 8.9.2 full prefill

如果：

- 没开缓存
- 没命中前缀
- 或策略要求回退

就执行：

```python
logits, past_key_values = forward_tokens(prefill_token_ids, past_key_values=past_key_values)
```

本质上就是标准 full prefill。

### 8.9.3 legacy KV-diff

流程是：

1. 对新 prompt 再做一次完整 full prefill，拿到 `new_past_key_values`
2. 对 `old V` 和 `new V` 做差，得到 token 分数
3. 用 `old KV` 做底座，把 selected 位置换成 `new KV`
4. 补算最后一个 prompt token logits

#### 理论问题

这条路线虽然“实现了选择性重算”，但它为了知道哪些 token 该重算，先完整算了一遍新 prompt。  
所以它的真实加速空间非常有限。

这也是代码注释里直接写明“可运行，但加速收益有限”的原因。

### 8.9.4 legacy Query-Aware

流程是：

1. 从旧缓存和新 prompt 的重叠区取候选 token
2. 用 query-aware 分数选位置
3. 把 selected token 打包前向，位置按原始索引给 `position_ids`
4. 把 packed KV scatter 回完整序列
5. 用 blended KV 补算最后一个 prompt token logits

它和 chunk packed 的近似本质相同：

- 有位置
- 但没有真实完整左上下文

因此它也是实验性的近似方案。

## 8.10 decode 阶段

所有路线最终都会汇聚到统一 decode 循环：

```python
for _ in range(req.max_new_tokens):
    next_token_id = self._sample_next_token(logits, req.temperature, req.top_p)
    logits, past_key_values = self.model_runner.forward_tokens(
        [next_token_id], past_key_values=past_key_values)
```

理论上：

- full prefill 的 attention 主成本在 prefill 阶段
- decode 每步只输入 1 个 token
- 有 KV cache 时，每步主要是“新 token 对历史长度 `L` 的注意力”

若生成长度为 `T`，上下文长度约为 `L`，则 decode 成本近似：

`O(T * L + T^2)`

在 `T << L` 时，可近似看作 `O(T * L)`。

## 8.11 缓存写回

生成结束后：

```python
self.kv_cache.put(
    req.session_id,
    final_token_ids,
    past_key_values,
    next_token_logits=next_token_logits,
)
```

这里写回的是：

- `prompt_token_ids + generated_ids`
- 对应的最终 `past_key_values`
- 如果本次没有生成 token，则额外缓存最后一步 `next_token_logits`

这意味着：

- warm/prefill 请求写回的是“prompt-only KV”
- 正常生成请求写回的是“prompt + generated tokens KV”

## 9. 复杂度与精确性总结

下面给一个更实用的比较表。设：

- `L`：总 prompt 长度
- `R`：可复用前缀长度
- `n`：chunk token 数
- `m`：query token 数
- `K`：QAW 选中 token 数
- `T`：生成 token 数

| 路径 | 主要前向成本 | 额外张量成本 | 精确性 |
|---|---|---|---|
| full prefill | `O(L^2)` | 较少 | 精确 |
| session exact prefix reuse | 增量部分约 `O((L-R)*R + (L-R)^2)` | 较少 | 精确 |
| chunk full reuse | miss chunk 的 `sum O(l_i^2)` | RoPE rebase + repeated concat，较重 | 近似 |
| legacy KV-diff | 先完整新 prefill `O(L^2)`，再选择性融合 | blended KV 拷贝 | 近似，且加速弱 |
| QAW packed | 打分 `O(n*m*d)` + packed 前向 `O(K^2)` | scatter `O(L)` | 近似 |
| QAW window | 打分 `O(n*m*d)` + `sum O(p_i*w_i + w_i^2)` | 多次 slice/scatter | 比 packed 更接近精确 |
| decode | `O(T*L + T^2)` | 较少 | 标准 KV decode |

### 9.1 当前实现最关键的近似假设

当前仓库的实验价值主要建立在以下假设之上：

1. **独立 chunk 的 KV 在经过 RoPE 重定位后，仍能作为全局长上下文的近似替代**
2. **只重算少量高风险 token，就能显著修复 chunk 近似带来的误差**
3. **query 和候选 token 的静态 input embedding 相似度，足以作为“高风险 token”选择信号**

这三条假设都不是 transformer 理论中的严格等价结论，它们是算法实验假设。

### 9.2 当前实现的工程性限制

从代码本身看，至少有以下限制：

1. **chunk full reuse 不是严格精确**  
   RoPE 只修正位置，不补回缺失左上下文。
2. **packed QAW 近似较强**  
   选中 token 重算时缺完整左邻居。
3. **KV-diff 需要一次完整新 prefill**  
   很难真正省算力。
4. **反复 `torch.cat` 和 `clone` 会造成显存带宽开销**
5. **缓存层只有单进程内对象管理，无分页、无异步搬运**

这些都值得在后续优化中单独处理。

## 10. 用 `blend_musique.py` 理解测试流程

`example/blend_musique.py` 是理解整个算法最好的入口之一，因为它把三条主要实验路线放到了同一条脚本里做对照。

## 10.1 实验目的

脚本比较三种路径：

1. `full_prefill`
2. `full_reuse`
3. `query_aware`

其中：

- `full_prefill` 是 baseline
- `full_reuse` 是 chunk KV 直接拼接
- `query_aware` 是在 chunk KV 上做选择性重算

## 10.2 Prompt 组织方式

脚本把 MusiQue 样本组织成：

- `PREFIX_PROMPT`
- 多个 passage 对应的 `doc_prompts`
- `QUERY_PROMPT + question`

构造函数有两个：

- `build_prefix_prompt(doc_prompts)`
- `build_final_prompt(doc_prompts, q_prompt)`

也就是说，实验里的 chunk 正是 passage 级文本块。

## 10.3 为什么先 warm chunk cache

脚本每条样本一开始会调用：

```python
warm_res = warm_chunk_cache(...)
```

它内部构造的是一个：

- `use_cache=True`
- `recompute_strategy="none"`
- `max_new_tokens=0`

的 chunk-aware 请求。

目的不是生成答案，而是：

> 让每个 passage 对应的 chunk KV 先进入 chunk cache

这样后面的 `full_reuse` 和 `query_aware` 才能测“复用命中后的真实行为”。

## 10.4 三条对照路径

### A. Baseline: full prefill

```python
use_cache=False
recompute_strategy="none"
```

这会直接进入 `_generate_chunk_aware()` 的 `not req.use_cache` 分支，对完整 prompt 做 full prefill。

### B. Full reuse

```python
use_cache=True
recompute_strategy="none"
```

这会：

1. 命中已经 warm 的 chunk cache
2. 构造 `prefix + rebased chunks`
3. 不做任何选择性重算
4. 直接进入 suffix prefill / decode

### C. Query-aware

```python
use_cache=True
recompute_strategy="query_aware"
recomp_ratio=args.qaw_ratio
qaw_type=args.qaw_type
```

这会：

1. 先拿到 chunk full reuse 上下文
2. 用 query 选出要重算的 chunk token
3. 根据 `qaw_type` 走 `packed` 或 `window`
4. 再 decode

## 10.5 指标采集

脚本记录三类指标：

1. **时延**
   - `first_token_latency_s`
   - `total_latency_s`
2. **质量**
   - `compute_f1(...)`
3. **缓存行为**
   - `chunk_cache_hits`
   - `chunk_cache_misses`
   - `recomputed_tokens`
   - `recompute_mode`

逐样本结果会写进 `sample_result`，全局平均写进 `run_summary`。

## 10.6 结果输出格式

脚本使用 `ExperimentOutputWriter` 把结果写到 `outputs/*.output`。

这意味着它是为后续批量分析准备的，而不是为终端交互准备的。  
常见分析流程通常是：

1. 跑脚本生成 `.output`
2. 用 R/Python 脚本汇总
3. 画 TTFT、总时延、F1 对比图

## 10.7 一次完整测试的实际顺序

对每条 MusiQue 样本，当前脚本的执行顺序是：

1. 读入样本
2. 生成 `doc_prompts` 和 `q_prompt`
3. 清空 `musique` 命名空间下的 chunk cache
4. `warm_chunk_cache()`
5. 跑 `full_prefill`
6. 跑 `full_reuse`
7. 跑 `query_aware`
8. 计算 F1
9. 写一条 `sample_result`
10. 清理 chunk cache、`gc.collect()`、必要时 `torch.cuda.empty_cache()`

这个流程很适合做单样本对照分析，因为三条路径共用同一条样本与同一份 chunk 切分。

## 10.8 推荐的运行方式

在项目环境中，示例命令可以写成：

```bash
python example/blend_musique.py \
  --model-name /root/models/Yi-6B \
  --count 30 \
  --qaw-ratio 0.7 \
  --qaw-type packed
```

如果想看 window 版本，把 `--qaw-type` 改成 `window` 即可。

## 11. 总结

当前仓库的核心思想可以概括成一句话：

> 先把长上下文拆成可缓存 chunk，复用独立 chunk 的 KV，再用少量选择性重算去修复这种近似复用带来的误差。

从实现上看，它已经具备以下研究价值：

- 有完整的 full prefill / full reuse / QAW 对照路径
- 同时支持 session 级复用和 chunk 级复用
- 有 RoPE 位置重定位
- 有 packed/window 两种 QAW 近似
- 有可直接跑实验的 `blend_musique.py`

但也必须明确：

- chunk full reuse 不是严格精确复用
- packed QAW 是进一步近似
- 当前实现的张量拼接与拷贝成本不低
- 更偏“研究原型”而非“生产推理系统”

如果后续要继续优化，最值得优先处理的方向通常是：

1. 减少 repeated `cat/clone/scatter` 的显存拷贝
2. 提升 QAW 打分信号质量
3. 提高局部重算的上下文保真度
4. 设计更接近真实 transformer 状态依赖的 chunk 表示
