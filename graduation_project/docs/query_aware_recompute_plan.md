# Query-aware 选择性重算方案（MVP）

## 1. 背景与问题
当前 `kv_diff` 策略要先拿到 `new KV` 再和 `old KV` 比较差异分数，因此会额外做一次 full prefill。

你问的关键点：
- Query-aware 能否避免这次 full prefill？

答案：
1. **理论上可以**：如果仅用 query/token embedding 相似度做选择，再用稀疏重算算子直接更新所选 token 的 KV。
2. **当前 MVP 实现仍保留 full prefill**：因为纯 Transformers 没有稀疏 KV 重算内核，先 full prefill 再融合最稳、最容易跑通。

---

## 2. 当前已实现的 Query-aware 策略
实现文件：`engine/inference_engine.py`

流程：
1. 对新 prompt 先做 full prefill，得到 `new_past_key_values`
2. 计算 query-aware 分数：
   - 取候选 token embedding 与 query token embedding 做 cosine
   - 按 token 取 `max(sim)` 作为相关性分数
3. 按 `recomp_ratio` 取 top-k token，并强制保留尾部 `suffix_len`
4. 用 `old KV` 做底座，仅在被选索引处替换为 `new KV`，得到 blended KV
5. 用 blended KV 继续 decode

---

## 3. 与 kv_diff 的差异
- `kv_diff`：分数来自 `V_new - V_old` 的 L2 偏差（数据驱动）
- `query_aware`：分数来自 Query-Token 语义相关性（语义驱动）

MVP 中两者都采用“先 full prefill 再融合”的执行框架，因此主要差异在“选哪些 token 重算”。

---

## 4. 参数建议
1. `recomp_ratio`: `0.1 ~ 0.2`
2. `suffix_len`: `24 ~ 64`（保证 query 尾段和答案前缀稳定）
3. `query_text`: 建议显式传入，避免系统模板噪声影响

---

## 5. 已在示例中接入
`example/blend.py` 现对比三种方法：
1. `kv_diff` 重算
2. `query_aware` 重算
3. `full_prefill` 基线

运行：
```bash
python example/blend.py
```

---

## 6. 后续（若追求真正省掉 full prefill）
需实现“稀疏 token 重算内核/流程”，让所选 token 能在不完整 full prefill 的情况下直接更新 KV；这超出当前 MVP 范围。
