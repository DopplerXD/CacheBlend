# Query-Aware 实验可视化图表说明

说明：图表标题、坐标轴和图例均使用英文，以规避论文绘图阶段中文字体显示异常；本说明文档使用中文记录用途。

- 数据来源：`/Users/doppler/Documents/Github/CacheBlend/outputs`
- 图表输出目录：`/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/qaw_visualizations/figures`
- 脚本路径：`/Users/doppler/Documents/Github/CacheBlend/scripts/visualize_qaw_experiments.R`
- 生成时间：`2026-04-26 00:43:35 CST`

## qaw-ratio 选择

主对比图使用的 qaw-ratio 为 `0.70`。选择规则：在覆盖 2 个模型 x 4 个数据集的非端点 ratio 中，先要求 Query-Aware 的平均 total latency speedup vs Full Prefill 不低于 `0.87`，避免选到接近 Full Prefill 的高成本端点；再选择平均 F1 retention 最高的 ratio。

- 平均 F1 retention：`0.9355`
- 平均 total latency speedup vs Full Prefill：`0.8721`
- 平均 total latency overhead：`20.24%`
- 平均 Query-Aware F1：`0.3448`
- 平均 Query-Aware total latency：`1.6792 s`

## 新绘制图表及作用

### 01. `fig01_best_ratio_latency_f1_grid.png`

- 英文图题：Latency-Quality Comparison at Selected QAW Ratio
- 图表类型：2 x 4 scatter grid
- 数据范围：Current root outputs/*.output, qaw-ratio 0.70
- 作用/意义：Compares total generation latency and F1 for the three methods across two models and four datasets.
- 论文使用建议：Main figure candidate for Chapter 4 method comparison.

### 02. `fig02_yi6b_ratio_metric_curves.png`

- 英文图题：Yi-6B QAW Ratio Curves
- 图表类型：Faceted line chart
- 数据范围：Yi-6B, four datasets, qaw-ratio 0.00 to 1.00
- 作用/意义：Shows how F1, total latency, and TTFT change with qaw-ratio while full reuse and full prefill stay as references.
- 论文使用建议：Good for explaining the ratio sensitivity of Yi-6B.

### 03. `fig03_qwen25_15b_ratio_metric_curves.png`

- 英文图题：Qwen2.5-1.5B QAW Ratio Curves
- 图表类型：Faceted line chart
- 数据范围：Qwen2.5-1.5B, four datasets, qaw-ratio 0.00 to 1.00
- 作用/意义：Shows how F1, total latency, and TTFT change with qaw-ratio while full reuse and full prefill stay as references.
- 论文使用建议：Good for explaining the ratio sensitivity of Qwen2.5-1.5B.

### 04. `fig04_yi6b_latency_quality_tradeoff_path.png`

- 英文图题：Yi-6B Latency-Quality Trade-off Path
- 图表类型：Trade-off scatter path
- 数据范围：Yi-6B query-aware curve with full reuse/full prefill reference points
- 作用/意义：Visualizes where each qaw-ratio sits in the latency-quality plane and whether it approaches the full prefill quality region.
- 论文使用建议：Useful for discussing the cost of recovering quality.

### 05. `fig05_qwen25_15b_latency_quality_tradeoff_path.png`

- 英文图题：Qwen2.5-1.5B Latency-Quality Trade-off Path
- 图表类型：Trade-off scatter path
- 数据范围：Qwen2.5-1.5B query-aware curve with full reuse/full prefill reference points
- 作用/意义：Visualizes where each qaw-ratio sits in the latency-quality plane and whether it approaches the full prefill quality region.
- 论文使用建议：Useful for discussing the cost of recovering quality.

### 06. `fig06_yi6b_ttft_total_quality.png`

- 英文图题：Yi-6B TTFT-Total Latency Structure
- 图表类型：TTFT-total scatter
- 数据范围：Yi-6B query-aware curve
- 作用/意义：Separates first-token delay from total generation latency and marks how quality changes along the curve.
- 论文使用建议：Useful when Chapter 4 discusses TTFT rather than only total latency.

### 07. `fig07_qwen25_15b_ttft_total_quality.png`

- 英文图题：Qwen2.5-1.5B TTFT-Total Latency Structure
- 图表类型：TTFT-total scatter
- 数据范围：Qwen2.5-1.5B query-aware curve
- 作用/意义：Separates first-token delay from total generation latency and marks how quality changes along the curve.
- 论文使用建议：Useful when Chapter 4 discusses TTFT rather than only total latency.

### 08. `fig08_f1_retention_heatmap.png`

- 英文图题：F1 Retention vs Full Prefill
- 图表类型：Heatmap
- 数据范围：Both models, four datasets, all qaw-ratios
- 作用/意义：Highlights which ratios preserve or exceed full prefill quality.
- 论文使用建议：Useful for selecting qaw-ratio and explaining dataset-specific quality behavior.

### 09. `fig09_total_latency_speedup_heatmap.png`

- 英文图题：Total Latency Speedup vs Full Prefill
- 图表类型：Heatmap
- 数据范围：Both models, four datasets, all qaw-ratios
- 作用/意义：Shows where query-aware recomputation saves total latency and where it becomes slower.
- 论文使用建议：Useful for latency analysis and negative-result discussion.

### 10. `fig10_balanced_score_heatmap.png`

- 英文图题：Balanced Quality-Latency Score
- 图表类型：Heatmap
- 数据范围：Both models, four datasets, all qaw-ratios
- 作用/意义：Combines quality retention and latency speedup into a compact ratio-selection view.
- 论文使用建议：Useful as supporting material for the selected qaw-ratio.

### 11. `fig11_f1_delta_against_baselines.png`

- 英文图题：Query-Aware F1 Delta against Baselines
- 图表类型：Delta line chart
- 数据范围：Both models, four datasets, all qaw-ratios
- 作用/意义：Shows whether query-aware recomputation improves over full reuse and how far it is from full prefill.
- 论文使用建议：Good for explaining quality recovery and degradation cases.

### 12. `fig12_latency_overhead_vs_prefill.png`

- 英文图题：Query-Aware Total Latency Overhead vs Full Prefill
- 图表类型：Overhead line chart
- 数据范围：Both models, four datasets, all qaw-ratios
- 作用/意义：Quantifies the latency cost of increasing recomputation ratio relative to full prefill.
- 论文使用建议：Useful for justifying the selected ratio and discussing runtime trade-offs.

### 13. `fig13_selected_ratio_normalized_summary.png`

- 英文图题：Selected Ratio Normalized Summary
- 图表类型：Grouped bar chart
- 数据范围：Both models at qaw-ratio 0.70
- 作用/意义：Summarizes quality retention, total latency ratio, and TTFT ratio against full prefill.
- 论文使用建议：Compact figure for reporting the chosen qaw-ratio.

### 14. `fig14_available_sample_metric_distributions.png`

- 英文图题：Available Sample-Level Metric Distributions
- 图表类型：Boxplot grid
- 数据范围：Fixed-ratio sample_result logs available in current outputs/*.output
- 作用/意义：Shows sample-level dispersion of F1, total latency, and TTFT for the three methods.
- 论文使用建议：Useful as auxiliary evidence for variance and outlier discussion.

### 15. `fig15_sample_delta_vs_full_prefill.png`

- 英文图题：Sample-Level Query-Aware Delta vs Full Prefill
- 图表类型：Sample scatter
- 数据范围：Fixed-ratio sample_result logs available in current outputs/*.output
- 作用/意义：Identifies samples where query-aware recomputation gains or loses quality and latency compared with full prefill.
- 论文使用建议：Useful for case analysis or error analysis subsection.

### 16. `fig16_recomputed_tokens_vs_latency.png`

- 英文图题：Recomputed Tokens vs Query-Aware Latency
- 图表类型：Sample scatter
- 数据范围：Fixed-ratio query-aware sample_result logs available in current outputs/*.output
- 作用/意义：Checks whether recomputation workload explains latency variation and whether higher workload aligns with quality changes.
- 论文使用建议：Useful as auxiliary analysis of runtime mechanism.

## 附带数据文件

- `qaw_curve_metrics.csv`：从 curve run_summary 解析出的宽表，含 F1、total_s、TTFT、retention、speedup 等派生指标。
- `qaw_ratio_scores.csv`：每个 qaw-ratio 在 8 个模型-数据集组合上的平均质量-延迟得分。
- `qaw_sample_metrics.csv`：若当前 outputs 根目录存在 sample_result，则保存样本级指标。
- `figure_inventory.csv`：图表清单的机器可读版本。
