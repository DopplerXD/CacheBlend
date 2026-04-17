# 图表清单

本清单由 `graduation_project/analyse/analyze_outputs_qaw_prefill_only.R` 自动生成。

## 分析规则

- 旧版 curve 输出文件会剔除文件内第一组 `run_summary`；带有独立 hot baseline 覆盖的新流程不再机械丢首组。
- 若存在匹配的 standalone hot baseline 文件，则优先用其覆盖目标文件中的 `full_prefill/full_reuse` 指标；`query_aware` 保持原输出。
- 主图和样本级图只展示 `full_prefill` 与 `query_aware`，不再将 `full_reuse` 纳入正文式对比。
- 只有在 `count` 和关键参数范围一致时，才允许把不同数据集放在同一张图中比较。
- `outputs/0416_sum/` 中的 orchestration 日志和失败/无指标文件不会进入正式分析。

## 全部图表

### fig_main_methods_blend_demo.png
- 标题：blend_demo 主实验方法对比
- 类型：Grouped Bar
- 数据范围：blend_demo 非曲线主实验
- 意义：比较 full_prefill 和 query_aware 在 TTFT、总时延和 F1 上的整体表现。
- 价值：适合做该数据集主结果概览，直接服务于论文主线的 prefill-vs-qaw 对比。

### fig_main_methods_musique.png
- 标题：musique 主实验方法对比
- 类型：Grouped Bar
- 数据范围：musique 非曲线主实验
- 意义：比较 full_prefill 和 query_aware 在 TTFT、总时延和 F1 上的整体表现。
- 价值：适合做该数据集主结果概览，直接服务于论文主线的 prefill-vs-qaw 对比。

### fig_main_methods_wikimqa.png
- 标题：wikimqa 主实验方法对比
- 类型：Grouped Bar
- 数据范围：wikimqa 非曲线主实验
- 意义：比较 full_prefill 和 query_aware 在 TTFT、总时延和 F1 上的整体表现。
- 价值：适合做该数据集主结果概览，直接服务于论文主线的 prefill-vs-qaw 对比。

### fig_ratio_curve_0416_blend_curve_default.png
- 标题：blend curve default 比例扫描曲线
- 类型：Line
- 数据范围：blend curve default
- 意义：展示重算比例变化对 query_aware 延迟与质量的影响，并与 full_prefill 基线对照。
- 价值：最适合解释参数敏感性和效率-质量权衡，是论文中参数分析的核心图。

### fig_ratio_curve_0416_musique_curve_025_100.png
- 标题：musique curve 025 100 比例扫描曲线
- 类型：Line
- 数据范围：musique curve 025 100
- 意义：展示重算比例变化对 query_aware 延迟与质量的影响，并与 full_prefill 基线对照。
- 价值：最适合解释参数敏感性和效率-质量权衡，是论文中参数分析的核心图。

### fig_ratio_curve_wikimqa_curve.png
- 标题：wikimqa curve 比例扫描曲线
- 类型：Line
- 数据范围：wikimqa curve
- 意义：展示重算比例变化对 query_aware 延迟与质量的影响，并与 full_prefill 基线对照。
- 价值：最适合解释参数敏感性和效率-质量权衡，是论文中参数分析的核心图。

### fig_ratio_heatmap_total_musique.png
- 标题：musique 总时延热图
- 类型：Heatmap
- 数据范围：musique 所有 ratio 曲线文件
- 意义：比较同一数据集下不同 ratio 曲线文件在各比例点上的 qaw 总时延分布。
- 价值：适合快速筛选低时延区间，也能看出不同曲线配置间的一致性。

### fig_ratio_heatmap_f1_musique.png
- 标题：musique F1 热图
- 类型：Heatmap
- 数据范围：musique 所有 ratio 曲线文件
- 意义：比较同一数据集下不同 ratio 曲线文件在各比例点上的 qaw 质量分布。
- 价值：适合寻找相对稳定的高质量区间，与时延热图结合可做参数筛选。

### fig_ratio_heatmap_total_wikimqa.png
- 标题：wikimqa 总时延热图
- 类型：Heatmap
- 数据范围：wikimqa 所有 ratio 曲线文件
- 意义：比较同一数据集下不同 ratio 曲线文件在各比例点上的 qaw 总时延分布。
- 价值：适合快速筛选低时延区间，也能看出不同曲线配置间的一致性。

### fig_ratio_heatmap_f1_wikimqa.png
- 标题：wikimqa F1 热图
- 类型：Heatmap
- 数据范围：wikimqa 所有 ratio 曲线文件
- 意义：比较同一数据集下不同 ratio 曲线文件在各比例点上的 qaw 质量分布。
- 价值：适合寻找相对稳定的高质量区间，与时延热图结合可做参数筛选。

### fig_tradeoff_musique.png
- 标题：musique 速度-质量权衡图
- 类型：Scatter Path
- 数据范围：musique ratio 曲线
- 意义：用速度提升和质量保持率同时观察不同比例点的综合收益。
- 价值：适合论文中解释“最优点不是单看延迟或单看F1，而是看两者平衡”。

### fig_tradeoff_wikimqa.png
- 标题：wikimqa 速度-质量权衡图
- 类型：Scatter Path
- 数据范围：wikimqa ratio 曲线
- 意义：用速度提升和质量保持率同时观察不同比例点的综合收益。
- 价值：适合论文中解释“最优点不是单看延迟或单看F1，而是看两者平衡”。

### fig_suffix_curve_0416_suffix_curve_musique_a.png
- 标题：suffix curve musique a 后缀长度扫描曲线
- 类型：Line
- 数据范围：suffix curve musique a
- 意义：展示 suffix_len 对 qaw 时延与质量的影响，用于判断尾部强制保留是否有必要。
- 价值：适合支撑“suffix_len 是稳定性控制参数”的论述。

### fig_suffix_curve_0416_suffix_curve_musique_b.png
- 标题：suffix curve musique b 后缀长度扫描曲线
- 类型：Line
- 数据范围：suffix curve musique b
- 意义：展示 suffix_len 对 qaw 时延与质量的影响，用于判断尾部强制保留是否有必要。
- 价值：适合支撑“suffix_len 是稳定性控制参数”的论述。

### fig_suffix_curve_0416_suffix_curve_wikimqa.png
- 标题：suffix curve wikimqa 后缀长度扫描曲线
- 类型：Line
- 数据范围：suffix curve wikimqa
- 意义：展示 suffix_len 对 qaw 时延与质量的影响，用于判断尾部强制保留是否有必要。
- 价值：适合支撑“suffix_len 是稳定性控制参数”的论述。

### fig_suffix_compare_cross_dataset.png
- 标题：跨数据集 suffix 对比
- 类型：Line
- 数据范围：仅使用 count 相同、固定 qaw_ratio 相同、suffix 范围相同的可比组
- 意义：比较 musique 与 wikimqa 在同一 suffix 扫描设置下的时延和质量走势。
- 价值：满足你的可比性约束，适合放在论文里说明参数对不同任务的影响是否一致。

### fig_suffix_tradeoff_cross_dataset.png
- 标题：跨数据集 suffix 权衡散点图
- 类型：Scatter Path
- 数据范围：跨数据集可比 suffix 组
- 意义：在相同 suffix 扫描设置下，比较两个数据集的速度-质量平衡曲线。
- 价值：适合用来解释“参数对不同任务的收益差异”。

### fig_sample_box_total_blend_demo.png
- 标题：blend_demo 样本级总时延箱线图
- 类型：Boxplot
- 数据范围：blend_demo 非曲线样本级结果
- 意义：展示不同方法在样本层面的总时延分布，而不是只看平均值。
- 价值：适合说明方法稳定性和离群样本情况。

### fig_sample_box_total_musique.png
- 标题：musique 样本级总时延箱线图
- 类型：Boxplot
- 数据范围：musique 非曲线样本级结果
- 意义：展示不同方法在样本层面的总时延分布，而不是只看平均值。
- 价值：适合说明方法稳定性和离群样本情况。

### fig_sample_box_f1_musique.png
- 标题：musique 样本级 F1 箱线图
- 类型：Boxplot
- 数据范围：musique 非曲线样本级结果
- 意义：比较 full_prefill 和 query_aware 在样本层面的质量分布差异。
- 价值：适合配合均值结果解释“平均值相近，但样本分布可能不同”。

### fig_sample_delta_scatter_musique.png
- 标题：musique 样本级增益散点图
- 类型：Scatter
- 数据范围：musique 非曲线样本级结果
- 意义：横轴是 query_aware 相对 full_prefill 的时延收益，纵轴是质量变化，四象限可直接解释收益与代价。
- 价值：非常适合论文分析“哪些样本既加速又不降质，哪些样本会退化”。

### fig_sample_latency_gain_hist_musique.png
- 标题：musique 时延收益直方图
- 类型：Histogram
- 数据范围：musique 非曲线样本级结果
- 意义：观察 query_aware 在样本层面的时延收益分布是否集中。
- 价值：适合补充说明收益是否普遍存在，还是主要由少数样本驱动。

### fig_sample_recomputed_vs_latency_musique.png
- 标题：musique 重算 token 数与时延关系图
- 类型：Scatter
- 数据范围：musique 非曲线样本级结果
- 意义：观察样本层面上重算 token 数量与 query_aware 实际时延之间的关系。
- 价值：适合支撑“重算规模会影响时延，但不是唯一因素”的分析。

### fig_sample_box_total_wikimqa.png
- 标题：wikimqa 样本级总时延箱线图
- 类型：Boxplot
- 数据范围：wikimqa 非曲线样本级结果
- 意义：展示不同方法在样本层面的总时延分布，而不是只看平均值。
- 价值：适合说明方法稳定性和离群样本情况。

### fig_sample_box_f1_wikimqa.png
- 标题：wikimqa 样本级 F1 箱线图
- 类型：Boxplot
- 数据范围：wikimqa 非曲线样本级结果
- 意义：比较 full_prefill 和 query_aware 在样本层面的质量分布差异。
- 价值：适合配合均值结果解释“平均值相近，但样本分布可能不同”。

### fig_sample_delta_scatter_wikimqa.png
- 标题：wikimqa 样本级增益散点图
- 类型：Scatter
- 数据范围：wikimqa 非曲线样本级结果
- 意义：横轴是 query_aware 相对 full_prefill 的时延收益，纵轴是质量变化，四象限可直接解释收益与代价。
- 价值：非常适合论文分析“哪些样本既加速又不降质，哪些样本会退化”。

### fig_sample_latency_gain_hist_wikimqa.png
- 标题：wikimqa 时延收益直方图
- 类型：Histogram
- 数据范围：wikimqa 非曲线样本级结果
- 意义：观察 query_aware 在样本层面的时延收益分布是否集中。
- 价值：适合补充说明收益是否普遍存在，还是主要由少数样本驱动。

### fig_sample_recomputed_vs_latency_wikimqa.png
- 标题：wikimqa 重算 token 数与时延关系图
- 类型：Scatter
- 数据范围：wikimqa 非曲线样本级结果
- 意义：观察样本层面上重算 token 数量与 query_aware 实际时延之间的关系。
- 价值：适合支撑“重算规模会影响时延，但不是唯一因素”的分析。

## 适合放入论文的图表建议

1. `fig_main_methods_musique.png`：musique 主实验方法对比
2. `fig_main_methods_wikimqa.png`：wikimqa 主实验方法对比
3. `fig_ratio_curve_0416_musique_curve_025_100.png`：musique curve 025 100 比例扫描曲线
4. `fig_ratio_curve_wikimqa_curve.png`：wikimqa curve 比例扫描曲线
5. `fig_tradeoff_musique.png`：musique 速度-质量权衡图
6. `fig_suffix_compare_cross_dataset.png`：跨数据集 suffix 对比
7. `fig_sample_delta_scatter_musique.png`：musique 样本级增益散点图

### 推荐放图逻辑顺序

1. 先放单数据集主实验方法对比图，交代全文的基线格局。
2. 再放核心 ratio 曲线，解释 query_aware 的参数敏感性。
3. 然后放速度-质量权衡图，突出“不是单看时延，也不是单看 F1”。
4. 接着放 suffix 对比图，说明尾部保留机制的作用。
5. 最后放样本级增益散点图，分析方法收益和退化案例的分布。

