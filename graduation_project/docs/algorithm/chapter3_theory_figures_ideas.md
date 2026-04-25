# 第三章算法合理性说明：图表、公式、伪代码与辅助代码设计

本文档面向本科毕业论文第三章“基于查询感知的 KV 选择性重算方法”，用于规划章节中的理论说明材料。重点不是实验结果分析，而是帮助读者理解算法为什么合理、流程如何运行、进入 Decode 前为什么还需要补算 logits，以及当前原型实现的边界。

建议第三章采用“核心图表 + 少量公式 + 两段伪代码 + 机制解释型辅助代码”的组合。图表不应承载 F1、TTFT、参数曲线等实验结论；如果使用单样本或 toy prompt，只用于解释机制，不作为性能证据。

---

## 0. 当前实现口径补充（Chunk KV Cache 版本，优先参考）

本节是对当前代码实现的补充说明，用于在保留旧图表素材的同时，明确论文第三章应优先采用的最新口径。后续旧素材中如果出现 session 级缓存、stale prompt、尾部强制重算等历史表述，写正文时应以本节为准进行替换或改写。

当前实现的主线已经从“完整 session prompt KV 复用”切换为“chunk 级 KV 缓存复用”。RAG 样本中的每一段 `text` 是一个 chunk。系统先对每个 chunk 独立 Prefill 并写入 chunk KV 池；推理时读取 chunk KV，通过 RoPE key 重定位将其从本地位置调整到完整 Prompt 的全局位置，再拼接为当前请求的上下文 KV。Query-aware 路径在 Full Reuse 的 KV 底座上选择部分与 query 更相关的 chunk token 做 packed Prefill，并将 selected KV scatter 回完整 KV。

### 0.1 第三章当前主线

建议正文围绕以下流程展开：

```text
RAG 文档重复出现
-> 每段 text 独立预热为 chunk KV
-> 推理时 RoPE 重定位并拼接 chunk KV
-> Full Prefill / Prefix Reuse / Full Reuse / Query-aware 路径对比
-> Query-aware 用 query-token embedding 相似度选择 chunk token
-> selected token packed Prefill
-> selected KV scatter 回完整 KV
-> suffix/query Prefill 后进入 Decode
```

核心表述：

- 缓存单位是一段 `text`，即一个 chunk。
- Full Prefill 不读取缓存，是标准质量基线。
- Prefix Reuse 用于说明传统前缀缓存机制和兼容路径，不是当前实验主线。
- Full Reuse 读取 chunk KV，做 RoPE key 重定位后直接拼接。
- Query-aware 在 Full Reuse 的 KV 底座上选择部分 chunk token 做 packed 重算。
- Packed 重算保留全局 `position_ids`，但仍是纯 Transformers 原型中的近似更新。
- 当前 chunk 主线中，query/suffix 会在 KV 拼接或融合后完整 Prefill，不再把尾部强制长度作为核心参数。

### 0.2 当前推荐图表

#### 图3-1 RAG 长上下文重复 Prefill 问题

图示内容：

- 多个 query 指向同一组或重叠的文档 chunk；
- 无缓存时，每个请求都对相同 chunk 重复 Prefill；
- 标出 Prefill 阶段是长上下文主要开销来源。

图注建议：

> RAG 场景下，不同查询可能反复召回相同文档片段，导致 Prefill 阶段重复构建相同文本块的 KV Cache。

#### 图3-2 三种推理路径对比

三栏结构：

| 路径 | 图中元素 |
| --- | --- |
| Full Prefill | 完整 Prompt 一次性进入模型，不访问 chunk KV 池 |
| Full Reuse | chunk KV 池 -> RoPE 重定位 -> KV concat -> suffix/query Prefill |
| Query-aware | Full Reuse 底座 -> query 打分 -> Top-K selected token -> packed Prefill -> scatter |

该图是第三章最重要的总览图。若需要引入 Prefix Reuse，可在图旁单独加一个小分支，说明其只在完整 token 前缀匹配时成立。

#### 图3-3 Chunk KV 预热与 RoPE 重定位

图示内容：

1. 每个 chunk 独立 Prefill，位置从 0 开始；
2. KV 池存储 `(K_i^{local}, V_i^{local})`；
3. 拼接时根据 chunk 在完整 Prompt 中的起始位置计算 global position；
4. key 执行：

```text
K_global = RoPE(global_pos) * RoPE(local_pos)^-1 * K_local
```

5. value 不经过 RoPE 旋转，可以直接复用。

图中应明确：RoPE 重定位解决的是位置一致性，不保证独立 chunk KV 与完整上下文 Prefill 严格等价。

#### 表3-1 主要符号表

建议保留以下符号：

| 符号 | 含义 |
| --- | --- |
| $C_i$ | 第 $i$ 个 chunk |
| $T_i$ | chunk token 序列 |
| $Q$ | query token 集合 |
| $P_{pre}$ | chunk 前的 prefix |
| $P_{suf}$ | chunk 后的 query/suffix |
| $(K_i^{local},V_i^{local})$ | 独立预热得到的 chunk KV |
| $(K_i^{global},V_i^{global})$ | RoPE 重定位后的 chunk KV |
| $(K_{reuse},V_{reuse})$ | Full Reuse 拼接 KV |
| $(K_{sel},V_{sel})$ | packed 重算得到的 selected KV |
| $(K_{blend},V_{blend})$ | Query-aware 融合 KV |
| $S$ | 被选中重算的 chunk token 集合 |
| $r$ | `recomp_ratio` |

#### 表3-2 方法路径对比表

| 方法 | 缓存单位 | 是否读取 chunk KV | 主要处理 | query/suffix 处理 | 作用 |
| --- | --- | --- | --- | --- | --- |
| Full Prefill | 无 | 否 | 完整上下文中全量 Prefill | 随完整 Prompt 一起 Prefill | 标准质量基线 |
| Prefix Reuse | 连续前缀 | 视情况 | 复用匹配前缀，只计算新增 token | 增量 Prefill | 传统缓存背景/兼容路径 |
| Full Reuse | chunk | 是 | RoPE 重定位后直接拼接 | 在拼接 KV 上完整 Prefill | 速度基线 |
| Query-aware | chunk | 是 | 先拼接，再 packed 重算 Top-K chunk token 并 scatter | 在融合 KV 上完整 Prefill | 质量/时延折中 |

#### 图3-4 Query-token 相似度矩阵

图示内容：

- 行：chunk token；
- 列：query token；
- 单元格颜色：embedding cosine similarity；
- 每行取最大值得到 `score_t`；
- 分数最高的 Top-K chunk token 被标为 selected。

公式：

```text
sim(t,j) = cos(e_t, u_j)
score_t = max_j sim(t,j)
```

#### 图3-5 Packed Prefill 与全局位置

图示内容：

- 从多个 chunk 中抽出 selected token；
- selected token 组成短 packed 序列；
- 每个 selected token 保留完整 Prompt 中的 global position；
- 模型前向输出 selected KV。

需要强调：

- packed 序列不是完整 Prompt；
- 全局 `position_ids` 保证位置编号一致；
- 该路径是原型中的近似重算。

#### 图3-6 KV Scatter 融合

图示内容：

- Full Reuse KV 是完整底座；
- selected KV 按 global position 写回；
- 未选中位置继续复用；
- 得到 blended KV；
- 在 blended KV 上继续 Prefill query/suffix 并进入 Decode。

公式：

```text
K_blend[i] = K_sel[i],   if i in S
K_blend[i] = K_reuse[i], otherwise

V_blend[i] = V_sel[i],   if i in S
V_blend[i] = V_reuse[i], otherwise
```

### 0.3 当前伪代码建议

#### 算法3-1 Query-aware chunk token 选择

```text
输入：chunk_tokens, chunk_global_positions, query_text, recomp_ratio
输出：selected_token_ids, selected_global_positions

1  query_tokens <- EncodeWithoutSpecialTokens(query_text)
2  chunk_emb <- Embedding(chunk_tokens)
3  query_emb <- Embedding(query_tokens)
4  chunk_emb <- L2Normalize(chunk_emb)
5  query_emb <- L2Normalize(query_emb)
6  sim <- chunk_emb * Transpose(query_emb)
7  score <- RowMax(sim)
8  topk_num <- max(1, floor(len(chunk_tokens) * recomp_ratio))
9  selected_chunk_indices <- TopK(score, topk_num)
10 selected_token_ids <- chunk_tokens[selected_chunk_indices]
11 selected_global_positions <- chunk_global_positions[selected_chunk_indices]
12 return selected_token_ids, selected_global_positions
```

#### 算法3-2 Query-aware KV 融合

```text
输入：reuse_past, selected_token_ids, selected_global_positions, suffix_tokens
输出：final_past, logits_for_decode

1  selected_past <- ForwardTokens(
       selected_token_ids,
       past_key_values = None,
       position_ids = selected_global_positions
   )
2  blended_past <- Scatter(
       base = reuse_past,
       updates = selected_past,
       indices = selected_global_positions
   )
3  logits_for_decode, final_past <- ForwardTokens(
       suffix_tokens,
       past_key_values = blended_past,
       position_ids = suffix_global_positions
   )
4  return final_past, logits_for_decode
```

### 0.4 当前实现涉及的主要参数

第三章写算法参数时，建议覆盖以下字段：

| 参数 | 作用 |
| --- | --- |
| `prefix_text` | 任务说明或文档区前导模板 |
| `chunk_texts` | 可缓存文本块列表，一段 `text` 对应一个 chunk |
| `suffix_text` | 当前 query、回答格式和生成入口 |
| `query_text` | Query-aware 打分使用的语义查询文本 |
| `chunk_namespace` | chunk KV 缓存命名空间 |
| `recompute_strategy` | 控制 `none` / `query_aware` 等路径 |
| `recomp_ratio` | 控制 Top-K chunk token 重算比例 |
| `qaw_type` | 当前实验主线按 packed 方式描述 |
| `max_new_tokens` | 最大生成长度 |
| `temperature` | 采样温度，实验通常为 0 |
| `top_p` | nucleus sampling 参数，实验通常为 1 |

### 0.5 当前写作中必须避免的旧表述

- 不要把当前实现写成会话级 KV 缓存或完整前缀缓存。
- 不要说 Query-aware 需要先 Full Prefill 得到 new KV。
- 不要说 packed 重算与 Full Prefill 严格等价。
- 不要把 query/suffix 写成复用缓存的一部分；它们是在 chunk KV 拼接或融合后完整 Prefill。
- 不要把历史 `kv_diff` 路径写成本文主方法。
- 不要把旧尾部强制重算长度写成当前 chunk 主线的核心参数。

---

## 1. 正文推荐组合

普通水平本科论文正文不宜放太多复杂图。建议正文采用以下组合：

| 编号 | 类型 | 名称 | 推荐位置 | 作用 |
| --- | --- | --- | --- | --- |
| 图 3-1 | 问题背景图 | RAG 长上下文重复 Prefill 问题图 | 3.1 | 说明为什么长文档 RAG 中存在重复计算 |
| 图 3-2 | 对比图 | Full Prefill / Full Reuse / Selective Recomputation 对比图 | 3.1 | 界定本文研究对象 |
| 图 3-3 | 主流程图 | Query-aware KV 选择性重算总体流程图 | 3.2 | 串起全章方法流程 |
| 图 3-4 | 机制图 | Query-token embedding 相似度打分图 | 3.3 | 解释 token 分数如何得到 |
| 图 3-5 | 衔接图 | 进入 Decode 前的 KV 与 logits 衔接图 | 3.4 | 解释 scatter 后为什么还要补算 logits |
| 表 3-1 | 符号表 | 主要符号定义表 | 3.1 或 3.3 | 降低公式理解成本 |
| 算法 3-1 | 伪代码 | 查询感知 Token 选择 | 3.3 | 描述 selected indices 的生成 |
| 算法 3-2 | 伪代码 | 选择性 KV 更新与 Decode 前准备 | 3.4 | 描述 packed prefill、scatter 和 logits 补算 |

正文中图表的叙事顺序建议为：

```text
重复计算问题
  -> 三种处理模式
  -> Query-aware 总体流程
  -> Embedding 相似度打分
  -> Token 选择集合
  -> Packed prefill + KV scatter
  -> Decode 前 logits 补算
  -> 方法边界
```

---

## 2. 扩展图表池

本节列出可选图表。并非全部都要放入正文，可根据论文篇幅选择。推荐正文优先使用“核心图表”，其余放附录或答辩材料。

### 2.1 问题定义类

#### 图 A1：RAG 重复文档复用示意图

用途：

- 解释多个 query 可能命中相同或相似文档 chunk；
- 说明 full prefill 会重复编码相同文档 token；
- 引出 KV Cache 复用和选择性重算的必要性。

建议画法：

```text
Query 1 -> [Doc A][Doc B][Question 1] -> Prefill
Query 2 -> [Doc A][Doc B][Question 2] -> Prefill
Query 3 -> [Doc A][Doc C][Question 3] -> Prefill

重复区域：Doc A / Doc B
变化区域：Question
```

图中颜色建议：

- 灰色：可复用文档 token；
- 橙色：每次变化的 query token；
- 红色边框：full prefill 中被重复计算的区域。

正文可写口径：

> 在长序列 RAG 场景中，多个请求可能共享大量文档前缀。如果每次都对完整 Prompt 执行 Prefill，就会对重复文档内容反复构建 KV Cache，从而带来冗余计算。

#### 图 A2：三种处理模式对比图

用途：

- 同时解释 `full_prefill`、`full_reuse`、`selective recomputation`；
- 帮助读者理解本文不是完全缓存系统，而是研究“哪些 token 值得更新”。

建议画成三栏：

| 模式 | 图中表现 | 解释 |
| --- | --- | --- |
| Full Prefill | 所有 token 都标为“重算” | 质量最稳，但开销最高 |
| Full Reuse | 所有旧 token 都标为“复用” | 开销最低，但上下文变化时可能失真 |
| Selective Recomputation | 少量 token 重算，其余复用 | 在速度和质量之间折中 |

图中颜色建议：

- 蓝色：复用 KV；
- 红色：重新计算 KV；
- 绿色：新增 token；
- 紫色：尾部强制保留 token。

#### 图 A3：计算开销直觉柱形示意图

用途：

- 说明本文目标是折中，不是宣称所有场景都最优；
- 不使用真实实验数据，只画概念关系。

建议画法：

```text
计算开销
Full Prefill:       ██████████
Selective Recompute:█████
Full Reuse:         ██
```

注意：

- 图注必须写明“示意图，非实验结果”；
- 不要标具体时间、F1、加速倍数。

### 2.2 Token 选择类

#### 图 B1：Query-token 相似度矩阵图

用途：

- 解释 query-aware 分数来自 embedding cosine；
- 展示候选 token 与 query token 之间存在可计算的相关性结构；
- 支撑公式 `score_i = max_j sim(i,j)`。

建议画法：

```text
                query token 1   query token 2   ...   query token m
candidate 1        0.10            0.30                 0.12
candidate 2        0.75            0.18                 0.20
candidate 3        0.05            0.68                 0.11
...
```

每一行取最大值：

```text
score_1 = max(0.10, 0.30, ..., 0.12)
score_2 = max(0.75, 0.18, ..., 0.20)
```

可直接复用：

- `scripts/visualize_query_aware_selection.py`
- `outputs/query_aware_figs/*similarity_heatmap.png`

#### 图 B2：Token score 曲线 + selected mask 图

用途：

- 解释 `recomp_ratio` 如何控制 Top-K 数量；
- 解释 `suffix_len` 为什么会强制选中尾部 token；
- 展示最终选择集合 `S` 的组成。

建议画法：

```text
score
  |
  |       /\          /\
  |  /\  /  \    /\  /  \
  |_/  \/    \__/  \/    \______
  +-------------------------------- token position
      ^     ^       ^      [suffix zone]
     Top-K selected         forced selected
```

图中标注：

- Top-K selected；
- forced suffix；
- added tokens；
- unselected reused tokens。

#### 图 B3：文本着色图

用途：

- 直接在原文中高亮被选 token；
- 适合答辩展示，让非系统方向老师也能理解“算法选择了和问题相关的词”。

建议画法：

```text
Passage: The [capital] of France is [Paris]. ...
Question: What is the capital of France?
```

颜色规则：

- 深红：query-aware Top-K；
- 紫色：suffix 强制区；
- 浅灰：未选中复用 token。

注意：

- 中文论文正文中不要贴过长文本；
- 选一个 toy prompt 或单样本片段即可。

#### 图 B4：Top-K 选择过程分层图

用途：

- 解释候选区间、suffix 区间和 added token 的关系；
- 说明 suffix 通常从 Top-K 候选中排除，避免重复计入。

建议画成三层：

```text
Layer 1: [candidate tokens................][suffix tokens][added tokens]
Layer 2: [Top-K candidate tokens selected ][forced suffix][forced added]
Layer 3: S = Top-K union suffix union added
```

适合放在 3.3.4 “Top-K 选择与尾部保留规则”。

#### 图 B5：Query-aware 与 random_topk 对照概念图

用途：

- 解释为什么设置 `random_topk` 消融变体；
- 不放实验数据，只解释 query-aware 选择不是随机抽样。

建议画法：

```text
Query-aware:
[相关文档片段] -> 多个 token 被选中

Random-topk:
[全文随机位置] -> token 分散被选中
```

正文口径：

> random_topk 不是本文方法，而是用于后续验证“基于查询相关性的选择”是否优于随机选择的对照变体。

### 2.3 KV 更新类

#### 图 C1：Packed Prefill 示意图

用途：

- 解释 selected token 如何从完整 prompt 中被抽出；
- 说明 selected token 保留原始 `position_ids`；
- 为后续 KV scatter 铺垫。

建议画法：

```text
Full prompt positions:
0  1  2  3  4  5  6  7  8
T0 T1 T2 T3 T4 T5 T6 T7 T8
   *     *        *     *

Packed selected sequence:
T1 T3 T6 T8
position_ids:
1  3  6  8
```

正文要强调：

- 打包后的序列变短；
- `position_ids` 保留原始位置；
- 但 packed prefill 不是 full prefill 的严格等价替代。

#### 图 C2：KV Scatter 融合图

用途：

- 解释 selected KV 如何写回旧 KV 底座；
- 展示“未选中位置复用，选中位置更新”。

建议画法：

```text
old KV:      [old][old][old][old][old][old][old]
selected KV:      [new]    [new]         [new]
blended KV:  [old][new][old][new][old][old][new]
```

可配合公式：

```text
K_blend[i] = K_sel[i], i in S
K_blend[i] = K_old[i], i not in S
```

#### 图 C3：新旧 KV 序列长度对齐图

用途：

- 解释 `T_added` 为什么必须加入选择集合；
- 说明当 `old_len < new_len` 时，新 prompt 尾部位置没有旧 KV 可复用。

建议画法：

```text
old tokens: [0][1][2][3][4]
new tokens: [0][1][2][3][4][5][6]
                         added tokens
```

正文口径：

> 对于超出旧缓存长度的新增 token，系统无法从旧 KV 中读取对应表示，因此需要把这些位置强制纳入更新集合。

#### 图 C4：近似性说明图

用途：

- 主动说明当前方法是工程近似；
- 避免论文被质疑“打包 token 前向是否等价于完整上下文前向”。

建议画成左右对比：

```text
Full Prefill:
所有 token 在完整上下文中共同前向

Packed Prefill:
只取 selected token 打包前向，保留 position_ids，再 scatter 回完整 KV
```

必须标注：

```text
位置编号保留，但 selected token 无法完整访问未选 token 的真实上下文。
```

适合放在 3.4.5 或 3.5 前。

### 2.4 Decode 衔接类

#### 图 D1：Decode 前 logits 补算图

用途：

- 解释进入 Decode 前最容易混淆的一步；
- 说明 blended KV 只是缓存状态，还需要最后 prompt token 的 logits 才能采样第一个生成 token。

建议画法：

```text
blended KV for positions [0, ..., new_len-2]
              +
last prompt token at position new_len-1
              |
        forward once
              |
          logits
              |
   sample first generated token
              |
        decode loop
```

正文口径：

> 自回归解码需要以上一个位置的 logits 作为下一 token 的分布。完成 KV 融合后，系统还需要基于融合后的前缀 KV 补算最后一个 Prompt token 的 logits，才能进入逐 token Decode 阶段。

#### 图 D2：Prefill / Decode 时间线图

用途：

- 解释 query-aware 重算发生在 decode 之前；
- 帮助读者区分 prefill 阶段优化和 decode 阶段生成。

建议时间线：

```text
旧缓存读取 -> query-aware 选点 -> packed prefill -> KV scatter -> logits 补算 -> decode step 1 -> decode step 2 -> ...
```

图中可标注：

- 选择性重算属于 Prefill 前处理/补偿环节；
- Decode 仍按标准自回归循环执行。

#### 图 D3：past_key_values 形状变化图

用途：

- 技术说明，不建议作为核心正文图；
- 可放附录或答辩备用。

建议内容：

```text
old_past:
[num_layers, batch, num_heads, old_len, head_dim]

selected_past:
[num_layers, batch, num_heads, |S|, head_dim]

blended_past:
[num_layers, batch, num_heads, new_len, head_dim]
```

注意：

- HuggingFace 实际每层是 `(key, value)`；
- 常见张量形状是 `[batch, heads, seq_len, head_dim]`；
- 论文中可以简化成概念形状，避免陷入实现细节。

---

## 3. 符号表设计

建议在 3.1 或 3.3 前放一个符号表，降低阅读门槛。

| 符号 | 含义 |
| --- | --- |
| `P_old` | 旧请求 Prompt |
| `P_new` | 当前请求 Prompt |
| `T_old` | 旧 Prompt token 序列 |
| `T_new` | 当前 Prompt token 序列 |
| `L_old` | 旧 token 序列长度 |
| `L_new` | 当前 token 序列长度 |
| `L_overlap` | 新旧 token 可对齐处理的重叠长度 |
| `Q` | 查询 token 集合 |
| `e_i` | 第 `i` 个候选 token 的输入 embedding |
| `u_j` | 第 `j` 个 query token 的输入 embedding |
| `score_i` | 第 `i` 个候选 token 的 query-aware 分数 |
| `S` | 最终待更新 token 索引集合 |
| `T_topk` | 按 query-aware 分数选出的 Top-K token 集合 |
| `T_suffix` | 尾部强制更新 token 集合 |
| `T_added` | 新 prompt 中旧缓存没有覆盖的新增 token 集合 |
| `K_old, V_old` | 旧 KV Cache |
| `K_sel, V_sel` | selected token packed prefill 得到的新 KV |
| `K_blend, V_blend` | 融合后的 KV Cache |
| `recomp_ratio` | Top-K 重算比例 |
| `suffix_len` | 尾部强制更新 token 数 |

---

## 4. 公式设计

### 4.1 选择性重算目标

可写为：

```text
min C(S),  s.t. Q(S) ≈ Q_full
```

含义：

- `C(S)` 表示选择集合 `S` 后的计算开销；
- `Q(S)` 表示选择性更新后的生成质量；
- `Q_full` 表示 full prefill 质量参考；
- 目标是在质量尽量接近 full prefill 的前提下减少重算 token。

论文口径：

> 该目标不是要求选择性重算与全量重算严格等价，而是在计算开销和质量保持之间取得折中。

### 4.2 查询 token 来源

```text
Q =
  Encode(query_text),              if query_text is provided
  Tail(P_new, suffix_len),          otherwise
```

解释：

- 优先使用显式 query，减少 prompt 模板噪声；
- 没有显式 query 时，用 prompt 尾部近似，因为 RAG prompt 通常把问题放在文档之后。

### 4.3 Embedding 相似度

```text
sim(i,j) = (e_i · u_j) / (||e_i|| ||u_j||)
```

其中：

- `e_i` 是候选 token 的输入 embedding；
- `u_j` 是 query token 的输入 embedding；
- cosine similarity 衡量两者方向相似度。

### 4.4 Token 分数聚合

```text
score_i = max_j sim(i,j)
```

解释：

- 一个候选 token 只要与 query 中某个关键词高度相关，就应有机会被选中；
- `max` 比 `mean` 更强调局部关键词匹配，适合问答场景。

### 4.5 选择集合

```text
S = T_topk ∪ T_added ∪ T_suffix
```

解释：

- `T_topk`：query-aware 高分 token；
- `T_added`：新增 token，无法从旧 KV 复用；
- `T_suffix`：尾部强制更新，保护 query 附近上下文。

### 4.6 KV 融合

```text
K_blend[i] =
  K_sel[i],   if i in S
  K_old[i],   otherwise
```

```text
V_blend[i] =
  V_sel[i],   if i in S
  V_old[i],   otherwise
```

注意：

- 严格实现中 `K_sel, V_sel` 是 packed 序列，需要按 `selected_indices` scatter 回原位置；
- 论文公式可以用上述分段形式表达融合直觉。

---

## 5. 伪代码设计

### 算法 3-1：查询感知 Token 选择

```text
Algorithm 3-1 Query-aware Token Selection

Input:
  old_tokens, new_tokens
  query_text
  recomp_ratio
  suffix_len

Output:
  selected_indices

1: overlap_len <- min(len(old_tokens), len(new_tokens))
2: if query_text is not empty then
3:     query_tokens <- EncodeWithoutSpecialTokens(query_text)
4: else
5:     query_tokens <- Tail(new_tokens, suffix_len)
6: end if

7: cand_tokens <- new_tokens[0 : overlap_len]
8: cand_emb <- Embedding(cand_tokens)
9: query_emb <- Embedding(query_tokens)
10: cand_emb <- L2Normalize(cand_emb)
11: query_emb <- L2Normalize(query_emb)

12: sim <- cand_emb * Transpose(query_emb)
13: score <- RowMax(sim)

14: suffix_start <- max(0, len(new_tokens) - suffix_len)
15: candidate_end <- min(overlap_len, suffix_start)
16: topk_num <- max(1, floor(candidate_end * recomp_ratio))
17: topk_indices <- TopK(score[0 : candidate_end], topk_num)

18: suffix_indices <- {suffix_start, ..., len(new_tokens)-1}
19: added_indices <- {overlap_len, ..., len(new_tokens)-1}
20: selected_indices <- Sort(topk_indices union suffix_indices union added_indices)
21: return selected_indices
```

正文解释重点：

- Top-K 候选区间排除 suffix，避免 suffix 重复占用比例；
- query-aware 默认不强制加入 changed token；
- `added_indices` 必须加入，因为旧缓存没有对应 KV。

### 算法 3-2：选择性 KV 更新与 Decode 前准备

```text
Algorithm 3-2 Selective KV Update and Decode Preparation

Input:
  old_past_key_values
  new_tokens
  selected_indices

Output:
  blended_past_key_values
  logits_for_decode

1: selected_token_ids <- new_tokens[selected_indices]
2: selected_position_ids <- selected_indices

3: selected_past <- ForwardTokens(
       selected_token_ids,
       past_key_values = None,
       position_ids = selected_position_ids
   )

4: base_past <- TruncateOrPad(old_past_key_values, len(new_tokens))
5: blended_past <- Scatter(base_past, selected_past, selected_indices)

6: if len(new_tokens) == 1 then
7:     logits_for_decode, final_past <- ForwardTokens(new_tokens[0], None)
8: else
9:     prefix_past <- Truncate(blended_past, len(new_tokens) - 1)
10:    last_token <- new_tokens[-1]
11:    logits_for_decode, final_past <- ForwardTokens(
          last_token,
          past_key_values = prefix_past,
          position_ids = [len(new_tokens) - 1]
       )
12: end if

13: return final_past, logits_for_decode
```

正文解释重点：

- `selected_past` 是 packed 序列上的 KV；
- `Scatter` 将 packed KV 写回原始 token 位置；
- Decode 需要 logits，因此 KV 融合后还要补算最后一个 prompt token。

---

## 6. 辅助代码设计

本节是可以实现的说明型脚本设计，不作为实验分析。优先复用已有 `scripts/visualize_query_aware_selection.py`。如果需要新增脚本，建议只新增一个轻量脚本 `scripts/trace_query_aware_mechanism.py`，负责导出 token 分数表和 shape trace。

### 6.1 Token 级打分导出脚本

用途：

- 输出一个样本中每个 token 的分数和选择状态；
- 支撑论文中的小样例表格或 selected mask 图。

输入：

```text
--prompt "..."
--query-text "..."
--recomp-ratio 0.7
--suffix-len 32
--output outputs/query_aware_figs/token_scores.csv
```

输出字段：

| 字段 | 含义 |
| --- | --- |
| `position` | token 位置 |
| `token_id` | token id |
| `token_text` | tokenizer 解码后的 token 文本 |
| `score` | query-aware 分数 |
| `is_topk` | 是否由 Top-K 选中 |
| `is_suffix` | 是否属于尾部强制区 |
| `is_added` | 是否属于新增区 |
| `is_selected` | 是否最终属于 `S` |

说明：

- 该表适合展示 10 到 30 个 token 的局部片段；
- 不要把它写成实验统计，只写成机制样例。

### 6.2 Similarity Heatmap 脚本

用途：

- 展示候选 token 与 query token 的 cosine similarity matrix；
- 说明 embedding 相似度是可计算矩阵，而不是抽象描述。

可复用：

```text
scripts/visualize_query_aware_selection.py
```

输出：

- `*_qaw_similarity_heatmap.png`
- `*_qaw_manifest.json`

图注建议：

> 颜色越深表示候选 token 与某个 query token 的 embedding 余弦相似度越高。本文方法对每个候选 token 取其与所有 query token 的最大相似度作为选择分数。

### 6.3 Selected Mask 可视化脚本

用途：

- 画出 token 位置维度上的 selected mask；
- 显示 Top-K、suffix、added token 三类来源。

输出图建议：

```text
token position -> selected source
```

颜色：

- 红色：Top-K；
- 紫色：suffix；
- 绿色：added；
- 灰色：unselected。

可以与 token score 曲线合并为一张图。

### 6.4 Decode 前 Trace 脚本

用途：

- 打印关键 shape，辅助解释进入 decode 前发生了什么；
- 不输出质量指标和时延指标。

建议输出：

```text
old_len: 1024
new_len: 1056
overlap_len: 1024
selected_count: 360
selected_past_len: 360
blended_past_len: 1056
prefix_past_len_for_logits: 1055
logits_shape: [1, 1, vocab_size]
```

解释重点：

- `selected_past_len` 与 `blended_past_len` 不同；
- `prefix_past_len_for_logits = new_len - 1`；
- `logits_shape` 说明已经得到进入 decode 的下一 token 分布。

### 6.5 Toy Prompt 演示脚本

用途：

- 构造短文本，避免真实数据太长；
- 用于画最清楚的机制示意图。

示例 prompt：

```text
Document:
Alice lives in Paris. Bob works in London. Carol studies in Berlin.

Question:
Where does Alice live?
```

预期解释：

- 与 `Alice`、`lives`、`Paris` 相关 token 分数更高；
- 与 Bob/London、Carol/Berlin 相关 token 理论上不应成为主要选择对象；
- 该例只说明机制直觉，不代表模型真实推理表现。

### 6.6 Embedding 最近邻说明脚本

用途：

- 对每个 query token 找最相似的候选 token；
- 辅助解释 `max_j` 聚合如何捕捉局部关键词匹配。

输出表：

| query token | top matched candidate token | position | similarity |
| --- | --- | --- | --- |
| Alice | Alice | 3 | 0.92 |
| live | lives | 4 | 0.81 |
| where | location-related token | 10 | 0.64 |

注意：

- 数值只用于单样本解释；
- 不要将其扩展为实验结论。

### 6.7 KV Shape 动画素材导出脚本

用途：

- 导出每一步 shape 和 selected indices；
- 后续可手工画成流程图或制作答辩动画。

输出建议：

```json
{
  "old_past_shape": "[layers, batch, heads, old_len, head_dim]",
  "selected_indices": [1, 3, 8, 12],
  "selected_past_shape": "[layers, batch, heads, selected_count, head_dim]",
  "blended_past_shape": "[layers, batch, heads, new_len, head_dim]",
  "logits_shape": "[batch, 1, vocab_size]"
}
```

---

## 7. 正文取舍建议

### 7.1 普通本科论文正文推荐

建议正文放：

1. RAG 重复 Prefill 问题图；
2. 三种模式对比图；
3. Query-aware 总体流程图；
4. Embedding 相似度打分图；
5. Decode 前 logits 补算图；
6. 符号表；
7. 两段伪代码。

这样已经足够支撑第三章的理论说明。

### 7.2 可选放正文补充

如果第三章篇幅允许，可以补充：

- Token score + selected mask 图；
- KV scatter 融合图；
- 近似性说明图。

其中 KV scatter 图和近似性说明图对技术严谨性很有帮助。如果只能二选一，优先放 KV scatter 图。

### 7.3 更适合附录或答辩

以下内容更适合作为附录或答辩展示：

- 文本着色图；
- toy prompt 示例；
- past_key_values shape trace；
- embedding 最近邻表；
- Query-aware 与 random_topk 概念对照图。

---

## 8. 第三章写作嵌入建议

### 3.1 问题定义

建议插入：

- 图 A1：RAG 重复文档复用示意图；
- 图 A2：三种处理模式对比图；
- 表 3-1：符号定义表。

要回答的问题：

- 为什么 full prefill 有冗余；
- 为什么 full reuse 可能不够可靠；
- 选择性重算要解决什么子问题。

### 3.2 方法总体思路

建议插入：

- 图 3-3：Query-aware 总体流程图。

要回答的问题：

- 方法整体是“选、算、融、接 decode”；
- 当前实现是纯 Transformers 原型，不是底层稀疏内核。

### 3.3 基于查询感知的 Token 选择

建议插入：

- 图 B1：Query-token 相似度矩阵图；
- 算法 3-1；
- 公式 `sim(i,j)`、`score_i`、`S`。

要回答的问题：

- query 表示从哪里来；
- token embedding 如何得到；
- 为什么取 max；
- `recomp_ratio` 和 `suffix_len` 分别控制什么。

### 3.4 选择性重算机制

建议插入：

- 图 C1：Packed Prefill 示意图；
- 图 C2：KV Scatter 融合图；
- 图 D1：Decode 前 logits 补算图；
- 算法 3-2。

要回答的问题：

- selected token 如何被打包；
- KV 如何写回旧缓存；
- 为什么 scatter 后不能直接 decode；
- 补算最后 prompt token logits 的作用是什么。

### 3.5 复杂度与边界

建议插入：

- 图 C4：近似性说明图，或只用文字解释；
- 复杂度公式 `O(L_c * m * d)`。

要回答的问题：

- query-aware 打分开销来自哪里；
- packed prefill 的收益和局限是什么；
- 为什么方法不保证与 full prefill 严格等价。

---

## 9. 必须避免的表述

1. 不要说当前实现已经完成 vLLM/PagedAttention 级别的稀疏重算内核。
2. 不要说 selected token packed prefill 与 full prefill 数学等价。
3. 不要把单样本可视化写成实验结论。
4. 不要把 query-aware 分数写成 attention 分数；当前实现是输入 embedding cosine。
5. 不要把 `full_reuse` 写成主要质量基线；第三章主要用于概念比较。
6. 不要把计算开销示意柱形图写成真实测量值。

---

## 10. 一句话总结

第三章的图表和理论说明应服务于一条主线：在 RAG 长上下文中，重复文档导致 Prefill 冗余；与当前 query 语义相关的 token 更值得更新；本文用 embedding cosine 构造 query-aware 分数，再通过 Top-K、suffix 和 added token 形成更新集合；当前原型用 packed prefill 与 KV scatter 完成近似选择性重算，并在进入 decode 前补算最后 prompt token logits。
