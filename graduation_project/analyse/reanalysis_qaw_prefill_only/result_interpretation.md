# 数据结果简要解读

本文档基于 [figure_inventory.md](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figure_inventory.md)、`summary_metrics_clean.csv` 和 `sample_metrics_clean.csv` 生成，面向论文正文或实验分析小节的写作。

## 1. 总体结论

当前这版分析只保留 `full_prefill` 与 `query_aware` 的主对比，不再把 `full_reuse` 作为核心结论对象。从现有结果看，`query_aware` 的主要价值已经比较清楚：在不少设置下，它能够明显降低 `TTFT` 和总时延，但不同数据集上的质量保持情况并不完全一致。

整体上可以概括为两点：

- 在 `musique` 和 `blend_demo` 上，`query_aware` 的加速效果比较明显，但质量并不总是优于 `full_prefill`，需要依赖合适的 `recomp_ratio` 与 `suffix_len` 做折中。
- 在 `wikimqa` 上，`query_aware` 的结果更积极，一些参数区间不仅延迟更低，而且 `F1` 还高于 `full_prefill`，说明查询感知选择在该数据集上更容易命中有效上下文。

因此，论文里的主结论可以写成：`query_aware` 方法已经表现出明确的时延收益，其质量表现与任务类型和参数设置有关；合理参数下可以实现“较低时延 + 可接受甚至更优质量”的折中。

## 2. 主实验解读

### 2.1 MusiQue

从主实验图看，`query_aware` 相比 `full_prefill` 有明显加速，但平均 `F1` 低于 `full_prefill`。这说明在多跳问答场景中，虽然查询感知重算能减少大量无关 token 的计算，但过于激进的选择会损失部分跨段推理所需的信息。

从比例扫描结果看，MusiQue 上存在比较明显的“速度-质量权衡”：

- 低比例时延最优。例如 `0416_musique_curve_025_100` 中，`recomp_ratio=0.30` 时，`query_aware` 总时延约为 `0.719s`，明显低于 `full_prefill` 的 `1.568s`，但对应 `F1` 只有 `0.135` 左右。
- 高比例更接近质量上界。例如 `recomp_ratio=0.85` 时，`query_aware` 的 `F1` 约为 `0.241`，已经接近甚至略高于 `full_prefill` 的 `0.240`，但此时时延优势已经明显减弱。
- 如果兼顾两者，`0.45` 左右是一个更自然的折中点。此时总时延约 `0.954s`，仍明显快于 `full_prefill`，同时 `F1` 约为 `0.195`，比低比例设置稳定得多。

因此，对 MusiQue 更稳妥的表述不是“全面优于基线”，而是“能显著降低时延，但需要通过中等重算比例控制质量退化”。

### 2.2 WikiMQA

WikiMQA 上的现象更有利于正文叙事。主实验结果显示，`query_aware` 的时延明显低于 `full_prefill`，且 `F1` 没有明显恶化。在部分曲线设置下，`query_aware` 的 `F1` 甚至高于 `full_prefill`。

比例扫描结果表明：

- `recomp_ratio=0.65` 时，总时延约 `1.237s`，低于 `full_prefill` 的 `1.715s`，且 `F1` 约为 `0.285`，高于 `full_prefill` 的 `0.249`。
- `recomp_ratio=0.70` 时，速度和质量都比较均衡：总时延约 `1.303s`，`F1` 约 `0.314`，是较适合正文强调的折中点。
- `recomp_ratio=0.85` 时，`F1` 进一步提高到约 `0.331`，但时延优势有所收缩。

这说明对于相对直接的信息匹配型任务，查询感知 token 选择更容易发挥作用。论文中可以把 WikiMQA 写成“方法有效性的正向案例”。

### 2.3 Blend Demo

`blend_demo` 更适合作为机制演示，而不是严格结论来源。结果显示它在时延上对 `query_aware` 很友好，例如主实验中 `query_aware` 的 `TTFT` 和总时延都明显低于 `full_prefill`。但该数据没有标准答案，缺乏 `F1` 指标，因此不宜在论文里承担“质量结论”的任务。

比较合适的用法是：

- 用它展示方法链路是可运行的；
- 用它说明低比例下时延收益可以非常明显；
- 不把它作为最终质量结论的主要依据。

## 3. suffix_len 解读

从 suffix 曲线看，`suffix_len` 主要起到“稳定尾部 query 区域”的作用，其影响在不同数据集上并不完全一致。

### 3.1 MusiQue

在两组 MusiQue suffix 实验中，趋势基本一致：

- `suffix_len` 从 `8` 增加到 `32` 时，总时延逐渐下降，从约 `1.50s` 降到约 `1.24s`。
- `F1` 不是单调变化，`24` 附近反而较低，`32` 时又有所回升。

这说明在 MusiQue 上，保留较长 query 尾部有利于控制选择错误，但中间区间并不一定稳定。论文里可以写成：`suffix_len` 对质量有调节作用，但其收益不是严格单调的，需要结合任务特点选取。

### 3.2 WikiMQA

WikiMQA 上的 suffix 现象更清晰：

- `suffix_len=24` 时，`F1` 约为 `0.381`，是几组里最高的。
- `suffix_len=32` 时，总时延约 `1.260s`，是几组里最快的，但 `F1` 回落到约 `0.314`。

因此，WikiMQA 上的 `suffix_len` 更像一个可调节的稳定性参数：

- 如果更重视质量，可以偏向 `24`；
- 如果更重视速度，可以偏向 `32`。

这类结果很适合在论文里写成“参数可用于控制速度与质量的偏好”。

## 4. 样本级现象

样本级散点图和直方图说明，`query_aware` 的收益并不是完全均匀分布的。

- 在 MusiQue 上，部分样本可以同时获得时延收益和较小质量损失，但也存在明显退化样本。这和多跳问答对跨段依赖更强是一致的。
- 在 WikiMQA 上，样本级分布更集中，说明该任务下查询感知选择更稳定。
- 重算 token 数与时延的散点图表明，两者总体相关，但不是线性对应关系，说明除了重算数量之外，token 的具体分布位置也会影响实际时延。

因此，论文中不应只写平均值，还应补一句：方法收益具有样本差异性，这也是后续改进 token 选择策略的重要方向。

## 5. 适合写进论文的结论口径

如果要写得稳，建议直接采用下面这套口径：

1. `query_aware` 相比 `full_prefill` 能显著降低预填充阶段相关时延，说明查询感知选择性重算思路是可行的。
2. 不同数据集的质量表现存在差异。WikiMQA 上方法更稳定，MusiQue 上则体现出更明显的速度-质量权衡。
3. `recomp_ratio` 是决定方法行为的核心参数：较低比例更快，较高比例更接近质量上界，中等比例通常提供更合理的折中。
4. `suffix_len` 可以作为辅助稳定参数，用于保护 query 尾部信息，但其收益不一定单调。
5. 因此，本文方法的主要贡献不是“在所有场景下全面优于基线”，而是“在合理参数下实现了可观的时延下降，并验证了查询感知 token 选择的有效性”。

## 6. 推荐引用图

如果正文只放少量图，建议按这个顺序引用：

1. [fig_main_methods_musique.png](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figures/fig_main_methods_musique.png)
2. [fig_main_methods_wikimqa.png](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figures/fig_main_methods_wikimqa.png)
3. [fig_ratio_curve_0416_musique_curve_025_100.png](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figures/fig_ratio_curve_0416_musique_curve_025_100.png)
4. [fig_ratio_curve_wikimqa_curve.png](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figures/fig_ratio_curve_wikimqa_curve.png)
5. [fig_tradeoff_musique.png](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figures/fig_tradeoff_musique.png)
6. [fig_suffix_compare_cross_dataset.png](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figures/fig_suffix_compare_cross_dataset.png)
7. [fig_sample_delta_scatter_musique.png](/Users/doppler/Documents/Github/CacheBlend/graduation_project/analyse/reanalysis_qaw_prefill_only/figures/fig_sample_delta_scatter_musique.png)

这套顺序基本对应“主结果 -> 参数分析 -> 权衡分析 -> 样本分析”的论文叙事。
