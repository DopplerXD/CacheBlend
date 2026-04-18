# 纯 Transformers 下 Query-aware 选择性重算算法原理（当前实现）

## 1. 文档目的

本文档描述当前仓库中“纯 Transformers + HF causal LM”版本的 Query-aware 选择性重算实现原理，覆盖：

- 触发条件与主链路
- 分数计算与 token 选择逻辑
- 打包重算与 KV 融合机制
- 与 full prefill / full reuse / kv_diff 的关系
- 复杂度、性能现象与局限性
- 实验脚本中的调用口径

对应核心代码：

- `engine/inference_engine.py`
- `model/hf_model.py`
- `cache/kv_cache.py`
- `schema/types.py`

---

## 2. 系统背景与输入输出

### 2.1 请求结构（`GenerateRequest`）

关键字段（`schema/types.py`）：

- `recompute_strategy`: `none | kv_diff | query_aware`
- `recomp_ratio`: 重算比例（Top-K 比例）
- `suffix_len`: 尾部强制重算长度
- `query_text`: query-aware 的语义来源文本
- `qaw_variant`: `default | no_suffix | random_topk`
- `qaw_random_seed`: random_topk 随机种子

### 2.2 结果结构（`GenerateResult`）

关键统计字段：

- `recompute_mode`: 实际执行模式（如 `query_aware_recompute`）
- `recomputed_tokens`: 本次重算 token 数
- `first_token_latency_s` / `total_latency_s`

---

## 3. 触发条件：什么时候会走 Query-aware

在 `InferenceEngine.generate()` 中，Query-aware **只有在以下条件同时满足时触发**：

1. `req.use_cache=True`
2. 当前 `session_id` 存在旧缓存
3. 新 prompt 与旧缓存 `token_ids` **不是前缀匹配**
4. `recompute_strategy == "query_aware"`

否则会走：

- 前缀命中：`cache_prefix_reuse`（增量 prefill 或直接复用 logits）
- 无缓存或不启用重算：`full_prefill`

因此，实验脚本里常用“先 warm 一个 stale prompt，再请求 final prompt”来人为制造“非前缀命中”，以强制进入 query-aware 分支。

---

## 4. Query-aware 的核心流程

入口函数：`_prefill_with_query_aware_recompute()`。

设：

- 旧缓存 token 序列为 `T_old`
- 新请求 token 序列为 `T_new`
- `overlap_len = min(len(T_old), len(T_new))`

### 步骤 1：构造 query token

- 若 `query_text` 非空：`Q = encode_no_special(query_text)`
- 否则：回退为 `T_new` 的尾段（长度约 `suffix_len`）

### 步骤 2：计算 token 语义相关性分数

对每个候选 token `i`（`i < overlap_len`）：

1. 取输入 embedding：`e_i`
2. 取 query token embedding 集合：`{q_j}`
3. L2 归一化后计算 cosine 相似度矩阵
4. 该 token 分数定义为：

\[
score_i = \max_j \; \cos(e_i, q_j)
\]

即“该 token 与任一 query token 的最大语义相似度”。

### 步骤 3：按规则选择重算索引 `S`

通用选择器 `_select_token_indices_from_scores()` 规则：

- `added`: 新增 token 索引（`len(T_old)` 之后）
- `forced`: 尾部强制重算区间（最后 `suffix_len` 个）
- `topk`: 在候选区间做 Top-K（比例 `recomp_ratio`）

候选区间会排除尾部强制区间，避免重复。

三种 QAW 子变体：

1. `default`
   - `S = added ∪ forced ∪ topk(score)`
2. `no_suffix`
   - `suffix_len = 0`
   - `S = added ∪ topk(score)`
3. `random_topk`
   - Top-K 不按分数，而是随机抽样（稳定种子）
   - `S = added ∪ forced ∪ random_topk`

注意：Query-aware 当前**不强制 changed token**（与 kv_diff 不同）。

### 步骤 4：打包重算（Packed Prefill）

将 `S` 对应的 token 抽出成短序列：

- `selected_token_ids = [T_new[i] for i in S]`
- `selected_position_ids = S`

然后做一次前向：

- `forward_tokens(selected_token_ids, position_ids=selected_position_ids, past=None)`
- 得到 `selected_past_key_values`

### 步骤 5：按位融合 KV（Scatter）

以旧 KV 为底座：

- 若旧长度不足新长度，先在尾部补零
- 将 `selected_past_key_values` 按索引 `S` scatter 回对应位置
- 得到融合后的 `blended_past_key_values`

### 步骤 6：补算最后一个 prompt token logits

- 截取 `blended_past_key_values` 到 `new_len - 1`
- 用最后一个 token 做一次前向，得到进入 decode 的 logits

返回：

- `logits`
- `past_key_values`
- `recomputed_tokens = |S|`

---

## 5. 与解码阶段的衔接

在 `generate()` 主流程中：

- Query-aware 返回后会设置 `prefill_token_ids=[]` 与 `recompute_mode="query_aware_recompute"`
- 当前版本已修复“重复补算最后 token logits”问题：若重算分支已经返回 `logits`，不会再额外补算一次

随后按标准 decode 循环逐 token 生成。

---

## 6. 与其他策略的关系

### 6.1 Full Prefill

- 无缓存或不使用重算时的基线
- 一次完整前向，结果最“真实”

### 6.2 Full Reuse

- 前缀完全匹配时，可直接复用 past
- 若缓存里有 `next_token_logits`，可 0 prompt 计算直接进入 decode

### 6.3 KV Diff

- 目标：按新旧 V 的 L2 偏差选 token
- 代价：要先做一次 full prefill 获取 `new KV` 再算差值
- 在纯 Transformers 下峰值显存和延迟压力通常更大

### 6.4 Query-aware（本文）

- 目标：用 query 语义引导重算 token
- 优点：不需要先 full prefill 得到 `new KV`
- 代价：有打分、选点、打包、scatter 等额外开销；近似误差更明显

---

## 7. 复杂度与性能直觉

设：

- 上下文长度 `L`
- 选中 token 数 `K=|S|`
- query token 数 `Q`
- hidden size `H`

粗略开销：

1. 打分：`O(L * Q * H)`（embedding cosine）
2. Top-K：约 `O(L log K)`（实现上由 `torch.topk`）
3. 重算：`O(K)` token 的一次前向（但仍含多层 attention 计算）
4. 融合：`O(K)` 索引 scatter（逐层）

因此并非“只要 K < L 就一定更快”。

- 当 `K` 较大（如 ratio 高、suffix 大）时，QAW 可能慢于 full prefill
- `no_suffix` 通常更快（因为 `K` 变小）
- 第一组/第一次调用常见冷启动抖动（编译/缓存/显存状态）

---

## 8. 算法近似性与理论局限

当前实现是“工程可运行的近似方案”，并非严格等价于全序列重算。

根因：

1. 被选 token 被打包成短序列单独前向，虽然传了原始 `position_ids`，但它们在该前向中无法访问未选 token 的真实上下文
2. 将 packed KV 直接 scatter 回完整位置，只在局部上“按位一致”，不是全局注意力一致

这会带来：

- 质量可能下降（F1/ROUGE 下降或波动）
- 随 ratio 增加的质量曲线不一定单调
- 速度收益与质量损失之间呈明显 trade-off

---

## 9. 当前实验口径建议

为了公平比较 `full_prefill` 与 `qaw_default`：

1. 固定相同数据、`count`、`max_new_tokens`
2. 固定 `qaw_ratio` 与 `suffix_len`
3. 区分冷启动与稳态（建议加 warmup）
4. 使用 `run_summary` 的方法级指标，不用整脚本总耗时

建议同时报告：

- `*_avg_ttft_s`
- `*_avg_total_s`
- 质量指标（QA 用 F1，SAMSum 用 ROUGE-L）
- `qaw_true_recompute_count`
- `recomputed_tokens` 分布（可选）

---

## 10. 与原始 CacheBlend 的差异（结论性说明）

当前版本在“思想上”保持了三段式：

1. 选（query-aware 选 token）
2. 重算（仅 selected token）
3. 融合（按位 scatter 回 KV）

但在执行形态上与依赖 PagedAttention/vLLM 的实现不同：

- 这里是纯 HuggingFace Transformers + 显式 Python 级 KV 操作
- 缺少底层块级缓存调度与高效 attention 内核协同
- 因此速度上更容易受额外算子和显存状态影响

这也是实验中会出现“某些设置下 query-aware 不快于 full prefill”的根本工程原因。

---

## 11. 术语对照

- stale cache：用于触发“非前缀命中”的旧缓存
- overlap：新旧 token 序列可对齐的前缀长度
- forced suffix：尾部强制重算区段
- packed prefill：仅对 selected token 进行一次打包前向
- scatter fusion：将 packed KV 按原索引写回完整 KV
