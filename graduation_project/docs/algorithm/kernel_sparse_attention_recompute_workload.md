# 从 Chunk QAW 原型走向内核级稀疏 Attention：工作量评估

本文档说明当前 Query-aware 选择性重算与真正内核级稀疏 Attention 之间的差距。当前项目主线是 chunk 级 KV 缓存：每段 `text` 独立预热为 chunk KV，推理时通过 RoPE key 重定位拼接，并在 Query-aware 路径中使用 packed Prefill 和 KV scatter 更新部分 chunk token。

结论先行：当前实现已经贯通“查询感知选择 -> selected token packed Prefill -> selected KV scatter -> suffix/query Prefill -> Decode”的原型流程，但它不是严格意义上的内核级稀疏 Attention。若要做到内核级精确重算，需要下沉到模型 attention 层或 CUDA/Triton kernel 层，工作量明显超过当前毕业设计原型范围。

## 1. 当前实现做到什么程度

当前 Query-aware 路径主要包括：

1. 读取或预热每个 chunk 的 KV；
2. 将 chunk key 从本地 RoPE position 重定位到完整 Prompt 的全局 position；
3. 拼接 `prefix KV + chunk KV`，得到 Full Reuse KV 底座；
4. 根据 `query_text` 与 chunk token embedding 的 cosine 相似度计算分数；
5. 按 `recomp_ratio` 选择 Top-K chunk token；
6. 抽出 selected token，构造 `selected_token_ids` 和全局 `selected_position_ids`；
7. 调用 HuggingFace 模型对 selected token 执行 packed Prefill；
8. 将 selected token 得到的新 KV scatter 回完整 KV 对应位置；
9. 在融合后的 past 上完整 Prefill query/suffix；
10. 进入标准 Decode。

可以概括为：

```text
chunk KV reuse
  -> RoPE rebase + KV concat
  -> query-aware selection
  -> selected token packed forward
  -> selected KV scatter
  -> suffix/query prefill
  -> decode
```

这个流程清晰、可运行、容易解释，也适合在论文中说明“基于查询感知选择的 KV 缓存更新”。

## 2. 为什么它还不是内核级稀疏 Attention

严格意义上的稀疏 Attention 重算，核心不只是“少 forward 一些 token”，而是：

```text
在完整上下文逻辑中，只更新 selected token 的 K/V，
同时 selected token 在重算时仍能正确 attend 到它们应当看到的历史上下文。
```

举一个简化例子。完整 Prompt 为：

```text
T0 T1 T2 T3 T4 T5 T6
```

如果只重算 `T3` 和 `T5`，理想的内核级稀疏 Attention 应该做到：

```text
重算 T3 时：能看到 T0 T1 T2 T3
重算 T5 时：能看到 T0 T1 T2 T3 T4 T5
只把 T3/T5 的新 K/V 写回 KV cache
其他位置继续复用旧 K/V
```

当前 packed Prefill 的处理更接近：

```text
抽出 selected token: T3 T5
作为短序列 forward: [T3 T5]
保留 position_ids: [3 5]
把得到的新 KV scatter 回原位置
```

虽然 position id 保留了原始位置，但 selected token 在这次短序列 forward 中无法完整访问未选中的上下文。因此，当前实现是“基于 packed 输入的选择性 KV 更新”，不是“完整上下文里的精确稀疏 Attention 内核重算”。

论文里建议使用以下命名：

- 查询感知选择性 KV 重算；
- 基于 chunk KV 缓存的选择性 Prefill；
- packed Prefill + KV scatter 的选择性缓存更新。

不建议直接称为：

- 精确稀疏 Prefill；
- kernel-level sparse attention；
- 与 Full Prefill 等价的稀疏重算。

## 3. 若要实现内核级精确重算，需要补齐什么

### 3.1 Attention 输入接口

当前 HuggingFace Causal LM 的常规 forward 假设输入 token 是一段连续序列，并配合 `past_key_values` 进行增量计算。内核级稀疏重算需要表达：

- 哪些 query 位置需要重算；
- 每个 query 位置对应完整上下文中的 global position；
- 每个 query 位置可以 attend 到哪些历史 key/value；
- 重算结果写回 KV cache 的哪些位置。

这需要比普通 `input_ids + position_ids + past_key_values` 更细的 attention metadata。

### 3.2 KV Cache 写回能力

当前 `cache/kv_fusion.py` 在 Python 层完成 selected KV scatter。内核级实现需要在 attention 或 cache 管理层支持按位置写回：

```text
cache[layer][head][global_position] = new_kv
```

这意味着 KV cache 不能只作为 append-only 的 decode past 使用，还要支持对已有位置的定点更新。

### 3.3 Mask 与位置编码

稀疏重算仍必须遵守 causal mask。第 `i` 个位置只能 attend 到 `<= i` 的上下文。同时，RoPE 位置必须与完整 Prompt 中的 global position 一致。对于 chunk KV 路径，还要保证预热 chunk key 已经从 local position 正确重定位到 global position。

### 3.4 调度与显存管理

内核级实现需要处理 selected position 的排序、批处理、KV 读写地址映射和显存复用。如果 selected token 很分散，调度开销可能抵消稀疏计算收益。因此，工程上还需要考虑：

- selected token 的排序和分组；
- 多层 KV 写回的一致性；
- 与 batch decode 的兼容性；
- cache eviction 和 chunk namespace 的关系。

## 4. 工作量评估

| 任务 | 难度 | 说明 |
| --- | --- | --- |
| 当前 packed 原型维护 | 低 | 已实现，主要做实验和文档同步 |
| Python 层 attention 改造 | 中 | 可验证思想，但速度收益有限 |
| HuggingFace attention 层定点更新 | 中高 | 需要改模型 forward 和 cache 结构 |
| vLLM/PagedAttention 级实现 | 高 | 涉及 block table、paged cache、kernel 和调度 |
| CUDA/Triton 自定义 kernel | 很高 | 超出当前原型范围 |

对毕业设计来说，当前 packed 原型足以支撑方法说明和实验对比。若后续继续研究，可以先在 Python/HuggingFace attention 层验证“selected token 在完整上下文下重算”的正确性，再考虑更底层的推理引擎优化。

## 5. 论文表述建议

可以强调：

- 当前方法解决的是 chunk KV 缓存条件下的查询感知选择性更新；
- RoPE 重定位是 chunk KV 拼接成立的必要条件；
- Query-aware 避免了为了选点而先执行完整 new KV 预计算；
- packed Prefill + scatter 是可运行的原型路径；
- 该路径不是严格等价于 Full Prefill 的内核级精确重算。

不需要在第三章展开底层 kernel 设计细节。第三章重点应放在 chunk 缓存、RoPE 重定位、query-aware 选择、packed 更新和 KV 融合这条可复现链路上。
