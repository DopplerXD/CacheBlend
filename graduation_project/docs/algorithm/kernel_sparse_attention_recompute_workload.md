# 从当前实现走向内核级稀疏 Attention 重算：所需工作与工作量评估

本文档从当前 CacheBlend 原型实现出发，说明如果要把现有 Query-aware 选择性重算进一步升级为“内核级稀疏 Attention 重算”，需要补齐哪些能力、会涉及哪些代码层次，以及工作量大概如何。

先给出结论：当前项目已经实现了“查询感知 Token 选择 + 稀疏输入 packed prefill + KV scatter 融合”的原型流程，但它还不是严格意义上的稀疏 Attention 重算。若要做到内核级稀疏 Attention，需要从 HuggingFace Python 层下沉到模型 attention 层甚至 CUDA/Triton kernel 层，工作量明显超过普通本科毕业设计的常规范围。

---

## 1. 当前实现做到什么程度

当前仓库的 Query-aware 路径主要位于 `engine/inference_engine.py`，核心流程是：

1. 根据当前 `query_text` 构造 query token；
2. 取候选 token 与 query token 的输入 embedding；
3. 用 cosine 相似度计算每个候选 token 的相关性分数；
4. 根据 `recomp_ratio`、`suffix_len` 和新增 token 构造 `selected_indices`；
5. 抽出 selected token，构造 `selected_token_ids` 和 `selected_position_ids`；
6. 调用 HuggingFace 模型对 selected token 执行一次 packed prefill；
7. 将 selected token 得到的新 KV scatter 回旧 KV 对应位置；
8. 用融合后的 KV 补算最后一个 prompt token 的 logits；
9. 进入标准 decode，并在请求结束后更新当前 session 的 KV cache。

可以概括为：

```text
Query-aware selection
  -> selected token packed forward
  -> selected KV scatter
  -> decode preparation
```

这个流程的优点是清晰、可运行、容易解释，也适合在本科论文里说明“查询感知选择性 KV 更新”的基本思想。它没有修改模型内部 attention 计算，也没有改写底层 CUDA/Triton kernel。

---

## 2. 为什么它还不是内核级稀疏 Attention 重算

严格意义上的稀疏 Attention 重算，核心不只是“少 forward 一些 token”，而是：

```text
在完整上下文逻辑中，只更新 selected token 的 K/V，
同时 selected token 在重算时仍能正确 attend 到它们应当看到的历史上下文。
```

举一个简化例子。完整 prompt 为：

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

当前 packed prefill 的处理更接近：

```text
抽出 selected token:
T3 T5

作为短序列 forward:
[T3 T5]

保留 position_ids:
[3 5]

然后把得到的新 KV scatter 回原位置。
```

虽然 position id 保留了原始位置，但 selected token 在这次短序列 forward 中无法完整访问未选中的 `T0/T1/T2/T4` 等上下文。因此，当前实现是“稀疏输入驱动的选择性 KV 重算”，不是“完整上下文里的稀疏 Attention 内核重算”。

论文里比较稳妥的命名是：

- 查询感知选择性 KV 重算；
- 基于稀疏输入的选择性 Prefill；
- packed prefill + KV scatter 的选择性缓存更新。

不建议直接称为：

- 稀疏 Attention 内核；
- kernel-level sparse attention；
- 精确稀疏 Prefill。

---

## 3. 内核级稀疏 Attention 重算应具备的能力

如果要把当前实现升级为真正的内核级稀疏 Attention 重算，至少要补齐以下能力。

### 3.1 稀疏重算位置描述

当前实现只有 `selected_indices`，它表示哪些 token 要更新。内核级实现还需要把这些索引转换成 attention kernel 能理解的调度信息，例如：

- 本次要重算哪些 query positions；
- 每个 selected position 对应的原始绝对位置；
- 每个 selected position 能 attend 到哪些 key/value positions；
- selected positions 在输出 KV cache 中写回哪里；
- batch 内不同样本 selected token 数不同如何处理。

仅有一维 `selected_indices` 不够，因为 attention 计算需要知道 query、key、value 的完整索引关系。

### 3.2 支持“稀疏 query，完整 key/value”的 Attention

真正的稀疏重算通常不是把 selected token 打包后只看 selected token，而是让 selected token 作为 query，去访问完整或部分完整的历史 key/value。

理想形式类似：

```text
Q_sparse = hidden_states[selected_indices]
K_context = hidden_states[0 : selected_position + 1]
V_context = hidden_states[0 : selected_position + 1]
Output_sparse = Attention(Q_sparse, K_context, V_context)
```

这要求 attention 层支持：

- query 维度是稀疏的；
- key/value 维度仍来自完整上下文或合法上下文窗口；
- causal mask 按 selected token 的真实位置生成；
- 输出只回写 selected positions。

HuggingFace 标准 `forward(input_ids=selected_token_ids, position_ids=selected_indices)` 并不会自动提供这种能力。

### 3.3 层间逐层更新

Transformer 的 KV 不是只在第一层产生一次。每一层的 selected token hidden state 都依赖上一层输出。若要做到更接近完整上下文的稀疏重算，需要逐层执行：

```text
layer 0:
  用旧上下文 + selected hidden states 计算 selected token 新 hidden states 和新 KV

layer 1:
  用 layer 0 更新后的 selected hidden states 继续计算

...

layer N:
  得到最终 selected token 的新 KV
```

这意味着不能只调用一次模型整体 `forward` 然后 scatter。更合理的做法是改模型层内部逻辑，让每一层都知道 selected positions，并在每一层完成局部 attention、MLP 和 KV 写回。

### 3.4 KV cache 原位写回

当前项目用 Python tensor clone/cat/scatter 构造新的 `past_key_values`。这很直观，但不一定高效。内核级实现更希望：

- 在预分配 KV cache 中原位更新 selected positions；
- 避免每层大量 clone；
- 避免反复构造完整新 KV；
- 与 decode 阶段的 KV cache layout 兼容。

如果使用 vLLM/PagedAttention，还会涉及 block table、slot mapping、paged cache 写入位置等概念；如果使用 HuggingFace，则要自己管理 tuple 形式的 `past_key_values`，性能收益会比较有限。

### 3.5 稀疏 Attention kernel 或框架支持

要真正获得内核级收益，需要 attention kernel 能处理稀疏 query 与不规则上下文。可能路线包括：

1. 改 HuggingFace 模型 attention forward；
2. 写 PyTorch 层面的 gather/scatter attention；
3. 写 Triton kernel；
4. 接入 vLLM/PagedAttention 并修改其 attention backend；
5. 基于 FlashAttention 的 varlen 或 block-sparse 能力做二次开发。

越靠近底层，性能潜力越大，但开发和调试成本也越高。

---

## 4. 从当前项目升级的可能路线

### 路线 A：保持 HuggingFace 层，实现“上下文窗口增强版”选择性重算

这是当前项目最现实的增强方向。做法不是直接写 kernel，而是在 selected token 前加入左上下文窗口，例如：

```text
selected seed: T10, T25
left_context_window = 8

实际重算:
T2..T10, T17..T25
```

优点：

- 不需要改模型内部；
- 仍可使用 HuggingFace 标准 forward；
- 能缓解孤立 token 缺少因果上下文的问题；
- 本科毕设或原型系统可以承受。

缺点：

- 仍不是内核级稀疏 Attention；
- 重算 token 数会增加；
- 重叠窗口需要合并，否则会重复计算；
- 对长上下文的性能收益有限。

工作量估计：

```text
1-3 天：实现窗口扩展、索引合并、文档更新
2-4 天：补充可视化和样本级验证
总计：约 3-7 天
```

### 路线 B：改 HuggingFace 模型 attention 层，实现稀疏 query 对完整 KV attention

这条路线仍在 Python/PyTorch 层，但需要进入模型结构内部。目标是让 selected token 的 hidden state 在每一层作为 sparse query，去 attend 到完整上下文 KV。

需要做的事：

- 找到具体模型的 attention 模块，如 Llama/Qwen/Yi 的 attention forward；
- 增加 `selected_indices`、`context_indices`、`cache_update_indices` 等参数；
- 每层 gather selected hidden states；
- 构造对应 causal mask；
- 用完整或部分 KV 计算 attention output；
- 让 selected hidden states 继续经过 residual、MLP、norm；
- 将每层 selected K/V 写回完整 KV cache；
- 保证 decode 仍能使用标准 past_key_values。

优点：

- 比 packed prefill 更接近完整上下文重算；
- 不必一开始写 CUDA/Triton；
- 更适合做研究原型。

缺点：

- 强依赖模型结构；
- 对不同模型兼容性差；
- Python/PyTorch gather/scatter 可能吞掉性能收益；
- attention mask 和 position embedding 容易出错；
- 单元测试复杂。

工作量估计：

```text
熟悉 HF 模型结构：2-4 天
实现单模型单卡原型：1-2 周
调试 KV/position/mask：1-2 周
质量与速度验证：1 周
总计：约 3-5 周
```

### 路线 C：接入 vLLM/PagedAttention，修改 attention backend

这条路线更接近原始 CacheBlend 方向。vLLM 已经有分页 KV cache、slot mapping、block table 和高效 attention backend。如果要做工程上更像样的稀疏 Attention 重算，可以考虑在 vLLM attention backend 中增加 selected token 的重算和写回逻辑。

需要做的事：

- 搭建可运行的 vLLM fork；
- 理解 model runner、attention metadata、paged KV cache、slot mapping；
- 在 prefill/check 阶段传入 selected indices；
- 修改 attention backend，使 selected query 能读取完整上下文 KV；
- 将 selected K/V 写回 paged cache 对应 slot；
- 保证 batch、chunk、prefix cache、decode 逻辑不被破坏；
- 增加 fallback 路径，出错时回退 full prefill。

优点：

- 更接近生产级推理系统；
- 有机会获得真实性能收益；
- 与 CacheBlend 类方法更一致。

缺点：

- 工程门槛高；
- vLLM 版本变动大；
- attention backend 调试困难；
- 需要 GPU 环境和较多压测；
- 对本科论文来说工作量偏大。

工作量估计：

```text
熟悉 vLLM 内部结构：1-2 周
跑通最小修改路径：2-4 周
实现 selected token attention + paged cache 写回：3-6 周
调试 batch/position/cache 边界：2-4 周
实验与文档：1-2 周
总计：约 2-4 个月
```

### 路线 D：自写 Triton/CUDA 稀疏 Attention kernel

这是最底层、也最重的路线。目标是写一个 kernel，让 sparse query positions 在完整上下文上做 causal attention，并将输出和 KV 写回指定位置。

需要做的事：

- 设计 sparse query 的输入布局；
- 设计 context index 或 block index；
- 支持 causal mask；
- 支持 batch 内不同 selected token 数；
- 支持 multi-head/GQA；
- 支持 RoPE 或模型位置编码；
- 写 forward kernel；
- 处理数值稳定性、softmax、dtype；
- 与 PyTorch/HF 模型对接；
- 做 correctness test，对比 full prefill 的 selected positions 输出；
- 做性能 benchmark。

优点：

- 如果成功，性能潜力最大；
- 最接近“内核级稀疏 Attention 重算”。

缺点：

- 实现难度最高；
- 调试成本极高；
- 很容易出现数值或 mask 错误；
- 需要较强 CUDA/Triton 背景；
- 不适合普通本科毕设主线。

工作量估计：

```text
Triton/CUDA 方案设计：1-2 周
最小 kernel 原型：2-4 周
RoPE/GQA/cache 写回适配：3-6 周
正确性和性能调试：4-8 周
总计：约 3-5 个月，且风险较高
```

---

## 5. 需要改动的接口与数据流

如果以路线 B/C 为目标，当前项目至少要扩展以下接口。

### 5.1 请求结构

当前 `GenerateRequest` 已有：

- `recompute_strategy`
- `recomp_ratio`
- `suffix_len`
- `query_text`
- `qaw_variant`

可能新增：

- `sparse_context_mode`: `packed | left_window | full_context_sparse_attention`
- `left_context_window`
- `preserve_head_tokens`
- `preserve_tail_tokens`
- `force_changed_for_qaw`
- `fallback_on_sparse_error`

这些字段不一定都要立即加入。若只是写论文，不需要实际扩展；若要做工程增强，建议先加入 `left_context_window` 和 fallback。

### 5.2 选择结果结构

当前代码只返回 `selected_indices`。内核级实现需要更丰富的结构，例如：

```text
selected_indices
context_indices_per_selected
position_ids
writeback_indices
attention_mask_spec
```

如果 batch 内每个样本 selected token 数不同，还需要：

```text
cu_selected_lens
cu_context_lens
max_selected_len
max_context_len
```

这些是 varlen/sparse kernel 常用的信息。

### 5.3 KV cache 写回接口

当前 Python 实现是：

```text
base_k[:, :, selected_indices, :] = selected_k
base_v[:, :, selected_indices, :] = selected_v
```

内核级实现更希望抽象成：

```text
write_selected_kv(cache, layer_id, selected_indices, new_k, new_v)
```

如果使用 paged cache，还需要：

```text
slot_mapping
block_table
cache_block_id
offset_in_block
```

---

## 6. 正确性验证需要怎么做

内核级稀疏 Attention 很容易“看起来能跑，但结果悄悄错”。验证至少要分三层。

### 6.1 位置与索引验证

检查：

- selected indices 是否升序；
- position ids 是否与原 prompt 位置一致；
- writeback 是否写到正确 token 位置；
- added token、suffix token 是否被正确处理；
- batch 内样本是否错位。

### 6.2 KV 对齐验证

构造短 prompt，对比：

```text
full prefill 得到的 K/V[selected_indices]
稀疏重算得到的 K/V[selected_indices]
```

如果采用完整上下文稀疏 attention，两者应尽量接近；如果采用 left window 或 packed prefill，则只能作为趋势参考，不能要求完全一致。

### 6.3 生成行为验证

用少量固定样本验证：

- 能否稳定进入 decode；
- logits shape 是否正确；
- past length 是否正确；
- decode 后 cache 是否继续增长；
- 结果是否存在明显乱码或提前 EOS。

性能指标应放在最后，因为正确性没稳之前，速度没有意义。

---

## 7. 工作量总体评估

| 路线 | 能否称为内核级稀疏 Attention | 工程难度 | 预计时间 | 适合当前论文吗 |
| --- | --- | --- | --- | --- |
| 当前 packed prefill + scatter | 否 | 低 | 已完成 | 适合作为主线 |
| 左上下文窗口增强 | 否 | 中低 | 3-7 天 | 适合作为可选增强 |
| HF attention 层 sparse query 改造 | 部分接近 | 中高 | 3-5 周 | 可作为展望或高阶扩展 |
| vLLM/PagedAttention backend 改造 | 接近 | 高 | 2-4 个月 | 不适合作为本科主线 |
| Triton/CUDA 自写 kernel | 是 | 很高 | 3-5 个月以上 | 不建议 |

对当前毕业论文而言，最稳妥的定位是：

```text
本文实现的是查询感知的选择性 KV 更新原型。
它通过稀疏输入 packed prefill 和 KV scatter 避免 Query-aware 路径中的完整 new KV 预计算。
内核级稀疏 Attention 重算需要进一步修改 attention 层或推理框架底层，
属于后续工程化优化方向。
```

---

## 8. 建议写进论文的表述

可以这样写：

> 当前原型系统没有直接改写底层 Attention 内核，而是采用 selected token packed prefill 与 KV 回填的方式实现选择性更新。该实现能够验证查询感知 Token 选择策略本身的可行性，也便于在纯 Transformers 环境中复现。若进一步追求更接近完整上下文的稀疏重算，需要让 selected token 在原始上下文位置上参与 attention，并将新 K/V 原位写回缓存；这通常要求修改模型 attention 层、vLLM/PagedAttention backend，甚至编写 Triton/CUDA kernel，工程复杂度显著高于本文原型系统。

也可以更口语化一点：

> 本文的实现更像是把“应该更新哪些 Token”这件事先跑通，而不是直接完成一个生产级稀疏 Attention 内核。真正的内核级实现需要解决 selected token 如何在完整上下文中重新 attention、KV 如何原位写回、不同样本稀疏长度如何批处理等问题。这些内容更适合作为后续工作展开。

---

## 9. 结论

从当前项目继续走向内核级稀疏 Attention 重算，主要难点不在 Token 选择，而在 Attention 计算和 KV Cache 写回的底层执行方式。

当前项目已经完成：

- query-aware token scoring；
- selected token selection；
- packed prefill；
- KV scatter；
- decode 前 logits 补算；
- session cache 更新。

内核级实现还需要：

- sparse query 对完整上下文 attention；
- 正确 causal mask；
- 每层 selected hidden state 递推；
- KV cache 原位写回；
- batch 内变长稀疏调度；
- 与 HF/vLLM attention backend 对接；
- 大量正确性和性能验证。

因此，内核级稀疏 Attention 重算更适合作为“未来工作”或“工程化扩展方向”。对于当前本科毕业论文，继续围绕查询感知选择性 KV 更新原型展开，是更稳、更可答辩的路线。
