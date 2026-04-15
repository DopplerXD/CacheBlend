# CacheBlend 当前实验脚本用法说明

本文档汇总当前仓库内所有实验相关脚本的用途、运行命令与输出说明。

## 1. 运行前准备

1. 激活环境并安装依赖：
```bash
source venv/bin/activate
pip install -r requirements.txt
```

2. 确认模型路径（默认读取 `config.py`）：
- 默认 `MODEL_NAME=/root/models/Yi-6B`
- 可临时覆盖：
```bash
export MODEL_NAME=/root/models/Yi-6B
export DEVICE=cuda
export MODEL_DTYPE=bfloat16
```

3. 所有实验日志默认写入 `outputs/*.output`。

---

## 2. 核心实验脚本（example/）

### 2.1 `example/blend.py`
- 作用：最小示例，对比 `full_prefill / full_reuse / kv_diff / query_aware`（固定 10 条样本，来自 `inputs/1.json~10.json`）。
- 运行：
```bash
python example/blend.py
```
- 输出：
  - 结构化结果：`outputs/YYYYMMDDHHMM_blend.output`
  - 控制台风格日志：`outputs/YYYYMMDDHHMM_blend_console.output`

### 2.2 `example/blend_musique.py`
- 作用：MusiQue 数据集评测，默认比较 `full_prefill / full_reuse / kv_diff / query_aware`。
- 数据：`inputs/musique_s.json`
- 当前默认样本数：30（脚本内固定 `if count == 30: break`）
- 运行：
```bash
python example/blend_musique.py
```
- 输出：`outputs/YYYYMMDDHHMM_musique.output`

### 2.3 `example/blend_wikimqa.py`
- 作用：WikiMQA 数据集评测。
- 数据：`inputs/wikimqa_s.json`
- 当前默认样本数：10（脚本内固定 `if count == 10: break`）
- 运行：
```bash
python example/blend_wikimqa.py
```
- 输出：`outputs/YYYYMMDDHHMM_wikimqa.output`

### 2.4 `example/blend_squad.py`
- 作用：SQuAD 子集评测。
- 数据：`inputs/squad_s.json`
- 运行：
```bash
python example/blend_squad.py
```
- 输出：`outputs/YYYYMMDDHHMM_squad.output`

### 2.5 `example/blend_musique_curve.py`
- 作用：MusiQue 上的 `query_aware` ratio 曲线评测（按 ratio 写多条 `run_summary`）。
- 当前 ratio 网格：`[0.40, 0.45, ..., 0.80]`（脚本中 `ratio_grid()`）
- 当前 stop count：50（脚本内 `stop_count = 50`）
- 运行：
```bash
python example/blend_musique_curve.py
```
- 输出：`outputs/YYYYMMDDHHMM_musique_curve.output`

### 2.6 `example/blend_runner.py`（推荐）
- 作用：统一 Runner，一次跑：
  - `full_prefill`
  - `full_reuse`
  - `kv_diff`
  - `qaw_default`
  - `qaw_no_suffix`
  - `qaw_random_topk`
- 支持数据集：`musique / wikimqa / squad`
- 常用命令：
```bash
python example/blend_runner.py --dataset musique --count 30 --qaw-ratio 0.3 --summary-only
python example/blend_runner.py --dataset wikimqa --count 50 --qaw-ratio 0.3
python example/blend_runner.py --dataset squad --count 100 --qaw-ratio 0.3 --suffix-len 32
```
- 主要参数：
  - `--dataset`：`musique|wikimqa|squad`
  - `--count`：样本数量上限
  - `--qaw-ratio`：qaw 重算比例
  - `--kvd-ratio`：kvd 重算比例
  - `--suffix-len`：尾部强制重算长度
  - `--qaw-random-seed`：`qaw_random_topk` 随机种子
  - `--summary-only`：只写 `run_summary`
- 输出：`outputs/YYYYMMDDHHMM_runner_<dataset>.output`

### 2.7 `example/blend_samsum.py`（旧版/依赖 vLLM）
- 作用：旧版 vLLM 路径实验（非当前 Transformers 主链路）。
- 依赖：`vllm`、`mistralai/Mistral-7B-Instruct-v0.2`
- 不建议在当前 MVP 主链路中优先使用。

---

## 3. 工具脚本（scripts/）

### 3.1 `scripts/preprocess_squad.py`
- 作用：下载/读取 SQuAD 原始数据并转为 `inputs/squad_s.json`。
- 常用命令：
```bash
python scripts/preprocess_squad.py --max-samples 1000
python scripts/preprocess_squad.py --input-json /path/to/train-v1.1.json --output-json inputs/squad_s.json --max-samples 1000
```

### 3.2 `scripts/summarize_outputs_to_csv.py`
- 作用：聚合多个 `.output` 中的结构化事件为 CSV。
- 默认聚合事件：`run_summary`
- 常用命令：
```bash
python scripts/summarize_outputs_to_csv.py
python scripts/summarize_outputs_to_csv.py --inputs "outputs/*musique*.output" --event run_summary --output outputs/musique_summary.csv
python scripts/summarize_outputs_to_csv.py --inputs "outputs/*.output" --event sample_result --output outputs/sample_result_aggregate.csv
```

### 3.3 `scripts/analyze_musique_degradation_cases.py`
- 作用：筛选 `full_prefill` 正确但 `qaw` 错误样本，并导出人工归因模板。
- 支持结果来源：
  - 统一 runner：`qaw_default / qaw_no_suffix / qaw_random_topk`
  - 旧脚本：`query_aware`
- 常用命令：
```bash
python scripts/analyze_musique_degradation_cases.py \
  --inputs "outputs/*musique*.output" \
  --qaw-key qaw_default \
  --top-k 20
```

单日志快捷模板：
```bash
python scripts/analyze_musique_degradation_cases.py \
  --single-log outputs/202604152042_musique.output \
  --qaw-key query_aware \
  --top-k 20 \
  --out-csv outputs/musique_degradation_cases_2042.csv \
  --out-jsonl outputs/musique_degradation_cases_2042.jsonl \
  --out-appendix-csv outputs/musique_degradation_appendix_2042.csv
```

- 输出文件：
  - `--out-csv`：常规分析表
  - `--out-jsonl`：逐条 JSON 便于二次处理
  - `--out-appendix-csv`：论文附录格式（含标签枚举）

---

## 4. 推荐实验流程（本科毕设）

1. 预处理数据（若需要）：
```bash
python scripts/preprocess_squad.py --max-samples 1000
```

2. 使用统一 Runner 跑主实验：
```bash
python example/blend_runner.py --dataset musique --count 100 --qaw-ratio 0.3 --summary-only
```

3. 聚合 run_summary：
```bash
python scripts/summarize_outputs_to_csv.py --inputs "outputs/*.output" --event run_summary --output outputs/run_summary_aggregate.csv
```

4. 退化案例分析：
```bash
python scripts/analyze_musique_degradation_cases.py --inputs "outputs/*musique*.output" --qaw-key qaw_default --top-k 20
```

