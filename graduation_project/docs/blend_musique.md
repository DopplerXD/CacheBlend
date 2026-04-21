# blend_musique.py 说明

MusiQue 主实验脚本用于在多文档问答数据集上比较 `full_prefill`、`full_reuse`、`query_aware` 三种路径，并把逐样本结果与整体汇总写入结构化日志。当前版本使用 chunk 级 KV 缓存：样本中的每段 `ctxs[*].text` 都作为独立 chunk 预热缓存。

## 数据格式

读取 json 数据，默认数据集路径为 `inputs/musique_s.json`。

- `question`：问题文本；
- `answers`：标准答案列表；
- `ctxs`：文档列表；
  - `title`
  - `text`

脚本逐条处理样本，并将 `ctxs[*].text` 转换为 `chunk_texts`。`question` 会进入 `suffix_text`，同时以 `query_text` 形式传给 query-aware 打分逻辑。

## 运行方式

```bash
python example/blend_musique.py
python example/blend_musique.py --count 50 --qaw-ratio 0.7
python example/blend_musique.py --output-dir outputs/useful
```

常用参数：

- `--count`：最多评测多少条样本；
- `--qaw-ratio`：`query_aware` 的 chunk token 重算比例；
- `--output-dir`：日志输出目录；
- `--model-name`：模型路径或 HuggingFace 名称；
- `--dataset`：MusiQue 数据文件路径。

## 运行流程

### 1. 初始化

脚本创建 `RuntimeConfig`，初始化：

- `HFModelRunner`
- `KVCacheManager`
- `InferenceEngine`
- `ExperimentOutputWriter`

启动后先写入一条 `run_start`，记录模型名、数据集路径、`count`、`qaw_ratio`、采样参数等信息。

### 2. 样本转换

每条样本会被整理成 chunk-aware 请求需要的三段结构：

```text
prefix_text + chunk_texts[] + suffix_text
```

- `prefix_text`：任务说明和文档区前导模板；
- `chunk_texts`：MusiQue 检索文档，每个 `text` 是一个 chunk；
- `suffix_text`：当前问题、回答格式和答案触发文本；
- `query_text`：当前问题本身。

### 3. Chunk 预热

正式比较三种方法前，脚本会先发起一次 `max_new_tokens=0` 的预热请求。引擎对每个 chunk 独立执行 prefill，并将 KV 写入 chunk KV 池。缓存 key 由 `chunk_namespace`、模型标识和 chunk token ids hash 构成。

预热得到的 KV 使用 chunk 本地 position。后续复用时，引擎会根据 chunk 在完整 prompt 中的全局位置对 key 做 RoPE 重定位，然后再拼接。

### 4. Full Prefill

`full_prefill` 不使用 chunk 缓存。它直接把 `prefix_text + chunk_texts[] + suffix_text` 拼成完整输入，一次性 prefill 后进入 decode。该结果作为标准质量基线。

关键设置：

- `use_cache=False`
- `recompute_strategy="none"`
- `chunk_texts` 仍用于构造完整输入，但不读取 chunk KV

### 5. Full Reuse

`full_reuse` 使用预热好的 chunk KV：

1. prefill `prefix_text`；
2. 读取每个 chunk KV；
3. 将 chunk key 从本地 RoPE position 重定位到完整 prompt 的全局 position；
4. 拼接 `prefix KV + chunk KV`；
5. 在拼接后的 past 上完整 prefill `suffix_text`；
6. decode。

它不重算 chunk token，因此通常 TTFT 最低。需要注意，RoPE 重定位保证位置编码一致，但独立 chunk KV 并不严格等价于完整上下文 prefill 下得到的 KV。

### 6. Query-aware

`query_aware` 在 `full_reuse` 的拼接 KV 底座上执行选择性更新：

1. 使用 `query_text` 与所有 chunk token 的输入 embedding 计算余弦相似度；
2. 按 `qaw_ratio` 选择 Top-K chunk token；
3. 将选中 token 打包成 packed 序列，并使用其全局 `position_ids` 执行 prefill；
4. 将 selected KV scatter 回完整 KV 的对应全局位置；
5. 在融合后的 past 上完整 prefill `suffix_text`；
6. decode。

该路径不需要先执行完整 prefill 得到 new KV，但 packed prefill 是原型系统中的近似重算，不应解释为与 full prefill 完全等价。

## 输出字段

逐样本日志包含：

- `sample_idx`
- `chunk_num`
- `question`
- `answers`
- `warm_chunk_cache_hits`
- `warm_chunk_cache_misses`
- `full_prefill`
- `full_reuse`
- `query_aware`

每种方法会记录：

- `generated_text`
- `ttft_s`
- `total_s`
- `recompute_mode`
- `recomputed_tokens`（主要由 query-aware 产生）
- `chunk_cache_hits`
- `chunk_cache_misses`
- `f1`

汇总日志 `run_summary` 包含：

- `sample_count`
- `full_reuse_avg_ttft_s`
- `query_aware_avg_ttft_s`
- `full_prefill_avg_ttft_s`
- `full_reuse_avg_total_s`
- `query_aware_avg_total_s`
- `full_prefill_avg_total_s`
- `full_reuse_avg_f1`
- `query_aware_avg_f1`
- `full_prefill_avg_f1`
- `qaw_true_recompute_count`
- `ended_at`

## 结果解读

正常情况下，时延顺序通常接近：

```text
full_reuse < query_aware < full_prefill
```

质量分数不一定单调。`full_prefill` 是标准基线；`full_reuse` 速度快但 chunk 表示是独立预热得到的；`query_aware` 会更新一部分高相关 chunk token，但 packed 重算仍是近似路径。因此在小样本或低重算比例下，F1 可能出现波动。
