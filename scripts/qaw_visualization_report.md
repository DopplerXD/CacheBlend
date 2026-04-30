# Query-Aware 实验可视化图表说明

说明：图表标题、坐标轴和图例均使用英文，以规避论文绘图阶段中文字体显示异常；本说明文档使用中文记录用途。

- 数据来源：`/Users/doppler/Documents/Github/CacheBlend/outputs`
- 图表输出目录：`/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/qaw_visualizations/figures`
- 脚本路径：`/Users/doppler/Documents/Github/CacheBlend/scripts/visualize_qaw_experiments.R`
- 生成时间：`2026-04-26 16:53:34 CST`

## qaw-ratio 选择

主对比图使用按模型分别设置的展示 ratio：`Yi-6B = 0.60`，`Qwen2.5-1.5B = 0.45`。

- 平均质量保持率：`0.8752`
- 平均 total latency speedup vs Full Prefill：`1.0525`
- 平均 total latency overhead：`-1.82%`
- 平均 Query-Aware 分数：`0.2561`
- 平均 Query-Aware total latency：`1.8260 s`

## 新绘制图表及作用

### 01. `fig17_best_ratio_ttft_f1_grid.png`

- 英文图题：TTFT-Quality Comparison
- 图表类型：2 x 3 scatter grid
- 数据范围：Current root outputs/*.output, model-specific selected settings
- 作用/意义：Compares TTFT and F1 or Rouge-L for the three methods across two models and three datasets.
- 论文使用建议：Useful as a companion figure to the total-latency comparison when Chapter 4 discusses first-token delay.

### 02. `fig18_selected_setting_score_bars.png`

- 英文图题：Selected Setting Score Bars
- 图表类型：Faceted bar chart
- 数据范围：Three datasets, Qwen and Yi under model-specific selected settings
- 作用/意义：Compares the selected-score levels of full reuse, query-aware, and full prefill across both models within each dataset.
- 论文使用建议：Useful for a compact side-by-side comparison of method quality under the chosen settings.

### 03. `fig02_yi6b_ratio_metric_curves.png`

- 英文图题：Yi-6B Quality Baseline Curves
- 图表类型：Faceted line chart
- 数据范围：Yi-6B, three datasets, score only
- 作用/意义：Shows how score changes with recompute ratio, with both Full Reuse and Full Prefill shown as references.
- 论文使用建议：Useful for discussing both quality recovery over reuse and the remaining gap to full prefill.

### 04. `fig03_yi6b_speed_baseline_curves.png`

- 英文图题：Yi-6B Speed Baseline Curves
- 图表类型：Faceted line chart
- 数据范围：Yi-6B, three datasets, TTFT only
- 作用/意义：Shows how TTFT changes with recompute ratio, with both Full Reuse and Full Prefill shown as references.
- 论文使用建议：Useful for discussing where Query-Aware sits between the fastest and the slowest first-token baselines.

### 05. `fig06_yi6b_ttft_total_quality.png`

- 英文图题：Yi-6B TTFT-Total Latency Structure
- 图表类型：TTFT-total scatter
- 数据范围：Yi-6B query-aware curve
- 作用/意义：Separates first-token delay from total generation latency and marks how quality changes along the curve.
- 论文使用建议：Useful when Chapter 4 discusses TTFT rather than only total latency.

### 06. `fig08_f1_retention_heatmap.png`

- 英文图题：Quality Retention vs Full Prefill
- 图表类型：Heatmap
- 数据范围：Both models, three datasets, all qaw-ratios
- 作用/意义：Highlights which ratios preserve or exceed full prefill quality.
- 论文使用建议：Useful for selecting qaw-ratio and explaining dataset-specific quality behavior.

### 07. `fig11_f1_delta_against_baselines.png`

- 英文图题：Query-Aware Score Delta against Baselines
- 图表类型：Delta line chart
- 数据范围：Both models, three datasets, all qaw-ratios
- 作用/意义：Shows whether query-aware recomputation improves over full reuse and how far it is from full prefill.
- 论文使用建议：Good for explaining quality recovery and degradation cases.

## 附带数据文件

- `qaw_curve_metrics.csv`：从 curve run_summary 解析出的宽表，含 score、total_s、TTFT、retention、speedup 等派生指标。
- `qaw_ratio_scores.csv`：每个 qaw-ratio 在 8 个模型-数据集组合上的平均质量-延迟得分。
- `qaw_sample_metrics.csv`：若当前 outputs 根目录存在 sample_result，则保存样本级指标。
- `figure_inventory.csv`：图表清单的机器可读版本。
