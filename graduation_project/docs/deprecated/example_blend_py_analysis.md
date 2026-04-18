# `example/blend.py` 作用详解

## 1. 脚本整体目标
`example/blend.py` 是一个最小复现脚本，用来对比两种推理路径：

- CacheBlend 路径：利用已收集历史 KV，并只重算部分 token
- 普通路径：完整 prefill（不做 CacheBlend 的 token 选择）

它会对 `inputs/1.json` 到 `inputs/10.json` 共 10 个样本循环执行，并打印：

- 生成文本（cached vs normal）
- TTFT（first token latency）对比

关键输出点在：`example/blend.py:98-106`。

## 2. 主流程分解

### 2.1 初始化模型与 tokenizer
- 加载 `mistralai/Mistral-7B-Instruct-v0.2`
- 显式设置 tokenizer
- 位置：`example/blend.py:6-10`

### 2.2 读取数据并切分
每个样本包含：
- 多个文档块 `0..chunk_num-1`
- 一个 query

脚本将文本转成 token id（去掉首 token）：`example/blend.py:21-22`。

### 2.3 构造“分块执行”输入格式
脚本用手工 token 前后缀拼接文档块与 query：
- `s_start_full`、`s_start`、`s_end`
- 文档块和 query 被组织为 `doc_chunk_ids`
- 位置：`example/blend.py:33-49`

### 2.4 第一阶段：逐块跑一遍并收集 KV
脚本先把 `collect=True`、`check=False`（`example/blend.py:51-52`），然后对每个 chunk 执行一次 `llm.generate(..., max_tokens=1)`。

每次执行后从每一层取：
- `llm_layers[j].self_attn.hack_kv`

并拼接成 `chunk_past_key_values`，最后写回：
- `model.old_kvs = chunk_past_key_values`

位置：`example/blend.py:57-80`。

这一步本质是在“离线准备历史 KV”（old_kv），供后续 CacheBlend 检查模式使用。

### 2.5 构造完整 prompt
把前面分块 token 重新拼成一个总输入 `input_ids`，再 decode 成 `input_prompt`。
- 位置：`example/blend.py:81-90`

### 2.6 第二阶段：CacheBlend 推理（带选择重算）
设置：
- `check=True`
- `collect=False`
- `suffix_len=last_len`

然后执行生成并记录 TTFT：`example/blend.py:93-99`。

这会触发底层“选择重要 token 重算”的路径（见下文第 3 节）。

### 2.7 第三阶段：普通完整 prefill 基线
再跑一次：
- `check=False`
- `collect=False`

记录 baseline 的生成结果和 TTFT：`example/blend.py:101-106`。

## 3. `blend.py` 如何触发底层 CacheBlend 逻辑
`blend.py` 本身不做 token 选择，它只负责把开关和数据喂给底层：

1. 通过 `cache_fuse_metadata` 控制模式（`check/collect/suffix_len`）
2. 通过 `model.old_kvs` 提供历史 KV
3. 在 `check=True` 的 prefill 里，底层会在检查层使用新旧 `V` 差分做 Top-K 选 token

实际实现位置：
- `vllm_blend/vllm/attention/backends/xformers.py:204-217`

## 4. 关键变量意义
- `collect`：是否采集当前运行产生的 K/V（写入 `hack_kv`）
- `check`：是否开启 CacheBlend 检查模式
- `old_kvs`：按层组织的历史 K/V 序列
- `suffix_len`：尾部强制重算 token 的长度
- `recomp_ratio`：前缀部分重算比例（默认 0.16，模型里定义）

## 5. 脚本中的注意点/潜在问题
1. `last_len = len([q_ids+s_end])` 实际恒为 `1`（因为对单元素 list 取 `len`），这和“query+suffix 的 token 数”语义可能不一致。
2. `num_layer = 32` 写死，依赖具体模型层数。
3. `s_start/s_end` 里的 token id 是手工硬编码，换模型/模板可能不适配。
4. 使用 `open(...)` 未显式关闭文件（可改为 `with open(...) as f:`）。

## 6. 一句话总结
`example/blend.py` 的核心作用是：先收集分块 KV，后在同一输入上分别跑“CacheBlend（部分重算）”和“完整 prefill”，对比生成结果与 TTFT，用于演示 CacheBlend 的推理加速效果。
