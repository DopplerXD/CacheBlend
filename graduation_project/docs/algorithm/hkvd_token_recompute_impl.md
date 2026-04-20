# CacheBlend 中“基于 HKVD 选择需要重算 Token”的实现定位

> 说明：本文档保留原始 CacheBlend/vLLM 路径的定位，同时补充当前仓库纯 Transformers 原型中的 `kv_diff` 对照实现。论文第三章应主要写 query-aware 方法，HKVD/kv_diff 只作为对照策略。

## 结论（先看这里）
在原始 CacheBlend 风格实现里，`HKVD` 没有作为显式字符串出现；对应实现可理解为“用新旧 `V` 的差分做 Top-K 选择重要 token”，核心代码曾定位在：

- `vllm_blend/vllm/attention/backends/xformers.py:204-217`

其中关键语句：

- `temp_diff = torch.sum((value[:-last_len,:,:]-value_old[:-last_len,:,:])**2, dim=[1,2])`
- `top_indices = torch.topk(temp_diff, k=topk_num).indices`
- 将 `top_indices` 与后缀 `last_indices` 拼接，作为 `imp_indices`（重要 token 索引）

这就是“基于 HKVD（可理解为历史 KV 差异）选择需重算 token”的实际落点。

当前仓库的主线已经切换为纯 Transformers 原型，对应的 `kv_diff` 对照实现位于：

- `engine/inference_engine.py::_compute_kv_diff_scores`
- `engine/inference_engine.py::_prefill_with_kv_diff_recompute`
- `engine/inference_engine.py::_build_blended_past_key_values`

当前 `kv_diff` 的核心逻辑仍然是对每层 `V_new - V_old` 的 L2 差异按 token 聚合，然后执行 Top-K 选择。

## 调用链（从入口到选择逻辑）

### 原始 vLLM/CacheBlend 路径

1. `example/blend.py` 打开 CacheBlend 检查模式：
   - 设置 `cache_fuse_metadata["check"] = True`，并写入 `suffix_len`（`example/blend.py:94-97`）。
2. `LlamaModel` 在 prefill 时进入“检查层”分支：
   - `vllm_blend/vllm/model_executor/models/llama.py:330-356`
3. 每层 attention 会收到 `status/cache_fuse_metadata/old_kv`：
   - `vllm_blend/vllm/model_executor/models/llama.py:158-188`
   - `vllm_blend/vllm/attention/layer.py:39-56`
4. 最终进入 xformers backend 的 `forward`，在 `status == 1` 时执行 token 选择：
   - `vllm_blend/vllm/attention/backends/xformers.py:204-217`

### 当前纯 Transformers 路径

1. 实验脚本用 `stale_prompt` 写入旧缓存；
2. `GenerateRequest(recompute_strategy="kv_diff")` 发起最终请求；
3. `InferenceEngine.generate()` 检测到旧缓存不是新 prompt 前缀，进入 `_prefill_with_kv_diff_recompute`；
4. `_prefill_with_kv_diff_recompute` 先对新 prompt 执行 full prefill，得到 `new_past_key_values`；
5. `_compute_kv_diff_scores` 计算每个 token 的 V 差异分数；
6. `_select_token_indices_from_scores(..., force_changed=True)` 选择 Top-K、changed token、新增 token 和尾部 token；
7. `_build_blended_past_key_values` 将选中位置的 new KV 写回 old KV 底座。

## 选择规则细化

### 原始 vLLM/CacheBlend 路径

在 `status == 1`（检查层）时，逻辑是：

1. 读取后缀长度：`last_len = cache_fuse_metadata['suffix_len']`（通常用于保证 query 末尾 token 必算）
2. 对前缀部分计算新旧 `V` 的 L2 差异分数：
   - `temp_diff[t] = ||V_new[t]-V_old[t]||_2^2`
3. 按 `recomp_ratio` 计算 Top-K 数量：
   - `topk_num = int((total_len-last_len) * recomp_ratio)`
4. 取前缀 Top-K，再把后缀全部并入：
   - `top_indices + last_indices`
5. 存入 `cache_fuse_metadata["imp_indices"]`，并仅保留这些 query 继续 attention。

### 当前纯 Transformers 路径

当前实现的分数为：

```text
score[t] = sum_layers ||V_new[layer, t] - V_old[layer, t]||_2^2
```

最终选择集合由以下部分组成：

```text
S = TopK(score, recomp_ratio) union ChangedTokens union AddedTokens union ForcedSuffix
```

其中 `ChangedTokens` 只在 `kv_diff` 中强制加入；query-aware 默认不强制加入 changed token。

## 选择后如何“重建完整 KV”
选择后并不是永久丢掉其余 token：

- 在 `status == 2` 阶段，代码用 `imp_indices` 把新算出的 `k/v` 回填到 `old_kv`，再恢复成完整 `key/value`：
  - `vllm_blend/vllm/attention/backends/xformers.py:240-245`

即：
- 检查层：只算重要 token（加速）
- 后续层：用“旧 KV + 更新后的重要位置”形成完整序列继续前向

## 相关控制参数
- `recomp_ratio`：重算比例，默认在模型里是 `0.16`
  - `vllm_blend/vllm/model_executor/models/llama.py:300-304`
- `suffix_len`：后缀强制重算长度
  - 由 `example/blend.py` 写入：`example/blend.py:96`

当前纯 Transformers 路径中，参数来自 `GenerateRequest`：

- `recompute_strategy="kv_diff"`
- `recomp_ratio`
- `suffix_len`
- `query_text` 可传入，但 `kv_diff` 分数本身不使用 query 语义。

## 补充说明
- 该实现目前强绑定在 `xformers` attention backend 分支中。
- `PagedAttention.write_to_paged_cache_blend` 虽已定义（`vllm_blend/vllm/attention/ops/paged_attn.py:83-98`），但当前主路径中实际调用的是 `write_to_paged_cache`（`xformers.py:268-272`）。

对论文第三章而言，HKVD/kv_diff 的主要作用是形成对比：

- `kv_diff` 回答“哪些位置的新旧状态差异大”；
- `query_aware` 回答“哪些位置与当前查询更相关”；
- 当前纯 Transformers 原型中，`kv_diff` 需要 full prefill 得到 new KV，而 query-aware 不需要先得到完整 new KV。
