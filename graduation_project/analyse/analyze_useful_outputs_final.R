#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
})

# =========================
# 1. 路径设置
# =========================
input_dir <- "outputs/useful"   # 修改成你的输出目录
fig_dir <- "graduation_project/analyse/final_figures"
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# =========================
# 2. 工具函数
# =========================
`%||%` <- function(a, b) {
  if (is.null(a) || length(a) == 0) b else a
}

safe_num <- function(x) {
  y <- suppressWarnings(as.numeric(x))
  if (length(y) == 0) return(NA_real_)
  y[1]
}

safe_parse_json <- function(line) {
  line <- trimws(line)
  if (!startsWith(line, "{")) return(NULL)
  out <- tryCatch(fromJSON(line, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(out) || is.null(out$event)) return(NULL)
  out
}

infer_dataset <- function(file_name, dataset_field = NULL) {
  x <- tolower(paste(file_name, dataset_field %||% "", collapse = " "))
  if (str_detect(x, "musique")) return("musique")
  if (str_detect(x, "wikimqa")) return("wikimqa")
  if (str_detect(x, "samsum"))  return("samsum")
  if (str_detect(x, "blend"))   return("blend_demo")
  return("unknown")
}

base_theme <- theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 15, hjust = 1),
    legend.position = "right"
  )

save_fig <- function(p, name, w = 10, h = 6) {
  ggsave(file.path(fig_dir, name), p, width = w, height = h, dpi = 300)
}

# =========================
# 3. 读取事件
# =========================
read_events_from_file <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  evs <- map(lines, safe_parse_json) %>% compact()

  run_start <- keep(evs, ~ identical(.x$event, "run_start"))
  run_start <- if (length(run_start) > 0) run_start[[1]] else NULL

  dataset <- infer_dataset(basename(path), run_start$dataset %||% "")
  map(evs, function(ev) {
    ev$source_file <- basename(path)
    ev$dataset_name <- dataset
    ev
  })
}

files <- list.files(input_dir, pattern = "\\.output$", full.names = TRUE)
if (length(files) == 0) stop("No .output files found.")

all_events <- map(files, read_events_from_file) %>% flatten()

run_summaries <- keep(all_events, ~ identical(.x$event, "run_summary"))
sample_results <- keep(all_events, ~ identical(.x$event, "sample_result"))

# =========================
# 4. 提取 run_summary
# =========================
extract_summary_rows <- function(ev) {
  methods <- c("full_reuse", "full_prefill", "query_aware", "kv_diff")
  metrics <- c("avg_ttft_s", "avg_total_s", "avg_f1", "avg_rouge_l")

  rows <- list()
  for (m in methods) {
    for (metric in metrics) {
      key <- paste0(m, "_", metric)
      if (!is.null(ev[[key]])) {
        rows[[length(rows) + 1]] <- data.frame(
          source_file = ev$source_file,
          dataset = ev$dataset_name,
          recomp_ratio = safe_num(ev$recomp_ratio %||% NA),
          method = m,
          metric = metric,
          value = safe_num(ev[[key]]),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (!is.null(ev$qaw_true_recompute_count)) {
    rows[[length(rows) + 1]] <- data.frame(
      source_file = ev$source_file,
      dataset = ev$dataset_name,
      recomp_ratio = safe_num(ev$recomp_ratio %||% NA),
      method = "query_aware",
      metric = "true_recompute_count",
      value = safe_num(ev$qaw_true_recompute_count),
      stringsAsFactors = FALSE
    )
  }

  bind_rows(rows)
}

summary_df <- map(run_summaries, extract_summary_rows) %>% bind_rows()

# =========================
# 5. 提取 sample_result
# =========================
extract_method_node <- function(ev, method, node) {
  if (is.null(node) || !is.list(node)) return(NULL)
  data.frame(
    source_file = ev$source_file,
    dataset = ev$dataset_name,
    sample_idx = safe_num(ev$sample_idx %||% NA),
    method = method,
    ttft_s = safe_num(node$ttft_s),
    total_s = safe_num(node$total_s),
    recomputed_tokens = safe_num(node$recomputed_tokens),
    f1 = safe_num(node$f1),
    rouge_l = safe_num(node$rouge_l),
    stringsAsFactors = FALSE
  )
}

sample_df <- list()
for (ev in sample_results) {
  for (m in c("full_reuse", "full_prefill", "query_aware", "kv_diff")) {
    if (!is.null(ev[[m]])) {
      row <- extract_method_node(ev, m, ev[[m]])
      if (!is.null(row)) sample_df[[length(sample_df) + 1]] <- row
    }
  }
}
sample_df <- bind_rows(sample_df)

method_levels <- c("full_reuse", "full_prefill", "query_aware")
summary_df$method <- factor(summary_df$method, levels = method_levels)
sample_df$method  <- factor(sample_df$method, levels = method_levels)

# =========================
# 6. 图1：主实验总对比图（2x2）
# =========================
main_df <- summary_df %>%
  filter(is.na(recomp_ratio), method %in% method_levels,
         metric %in% c("avg_total_s", "avg_ttft_s", "avg_f1", "avg_rouge_l")) %>%
  group_by(dataset, method, metric) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  filter(!is.na(value))

metric_label <- c(
  avg_total_s = "Average Total Latency (s)",
  avg_ttft_s  = "Average TTFT (s)",
  avg_f1      = "Average F1",
  avg_rouge_l = "Average ROUGE-L"
)
main_df$metric <- factor(main_df$metric, levels = names(metric_label), labels = metric_label)

p1 <- ggplot(main_df, aes(method, value, fill = method)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.25, size = 3) +
  facet_grid(metric ~ dataset, scales = "free_y") +
  labs(title = "Overall Performance Comparison", x = "Method", y = NULL) +
  base_theme +
  guides(fill = "none")

save_fig(p1, "fig1_overall_comparison.png", 12, 8)

# =========================
# 7. 图2：ratio-latency 曲线
# =========================
lat_curve_df <- summary_df %>%
  filter(!is.na(recomp_ratio),
         method %in% c("query_aware", "full_prefill"),
         metric == "avg_total_s") %>%
  group_by(dataset, recomp_ratio, method) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop")

p2 <- ggplot(lat_curve_df, aes(recomp_ratio, value, color = method)) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2) +
  facet_wrap(~dataset, scales = "free_y") +
  labs(
    title = "Latency vs Recompute Ratio",
    x = "Recompute Ratio",
    y = "Average Total Latency (s)"
  ) +
  base_theme

save_fig(p2, "fig2_latency_vs_ratio.png", 10, 5)

# =========================
# 8. 图3：ratio-quality 曲线
# =========================
quality_metric <- if ("samsum" %in% unique(summary_df$dataset)) "avg_rouge_l" else "avg_f1"

qual_curve_df <- summary_df %>%
  filter(!is.na(recomp_ratio),
         method %in% c("query_aware", "full_prefill"),
         metric == quality_metric) %>%
  group_by(dataset, recomp_ratio, method) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop")

p3 <- ggplot(qual_curve_df, aes(recomp_ratio, value, color = method)) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2) +
  facet_wrap(~dataset, scales = "free_y") +
  labs(
    title = "Quality vs Recompute Ratio",
    x = "Recompute Ratio",
    y = ifelse(quality_metric == "avg_f1", "Average F1", "Average ROUGE-L")
  ) +
  base_theme

save_fig(p3, "fig3_quality_vs_ratio.png", 10, 5)

# =========================
# 9. 图4：speed-quality trade-off
# =========================
trade_speed <- summary_df %>%
  filter(!is.na(recomp_ratio), method == "query_aware", metric == "avg_total_s") %>%
  select(dataset, recomp_ratio, latency = value)

trade_quality <- summary_df %>%
  filter(!is.na(recomp_ratio), method == "query_aware", metric == quality_metric) %>%
  select(dataset, recomp_ratio, quality = value)

trade_df <- left_join(trade_speed, trade_quality, by = c("dataset", "recomp_ratio"))

baseline_df <- summary_df %>%
  filter(is.na(recomp_ratio), method == "full_prefill",
         metric %in% c("avg_total_s", quality_metric)) %>%
  select(dataset, method, metric, value) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  rename(latency = avg_total_s,
         quality = !!quality_metric)

p4 <- ggplot(trade_df, aes(latency, quality, color = dataset)) +
  geom_path(linewidth = 1) +
  geom_point(aes(size = recomp_ratio), alpha = 0.9) +
  geom_point(data = baseline_df, aes(latency, quality),
             inherit.aes = FALSE, shape = 4, size = 4, stroke = 1.2, color = "black") +
  labs(
    title = "Speed–Quality Trade-off of Query-aware Recompute",
    x = "Average Total Latency (s)",
    y = ifelse(quality_metric == "avg_f1", "Average F1", "Average ROUGE-L"),
    size = "Ratio"
  ) +
  base_theme

save_fig(p4, "fig4_speed_quality_tradeoff.png", 8, 5)

# =========================
# 10. 图5：样本级时延分布
# =========================
p5 <- sample_df %>%
  filter(method %in% method_levels, !is.na(total_s)) %>%
  ggplot(aes(method, total_s, fill = method)) +
  geom_boxplot(outlier.alpha = 0.2) +
  facet_wrap(~dataset, scales = "free_y") +
  labs(
    title = "Sample-level Total Latency Distribution",
    x = "Method",
    y = "Total Latency (s)"
  ) +
  base_theme +
  guides(fill = "none")

save_fig(p5, "fig5_sample_latency_boxplot.png", 10, 5)

# =========================
# 11. 图6：重算token数与时延关系
# =========================
p6 <- sample_df %>%
  filter(method == "query_aware", !is.na(recomputed_tokens), !is.na(total_s)) %>%
  ggplot(aes(recomputed_tokens, total_s, color = dataset)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  labs(
    title = "Relationship Between Recomputed Tokens and Latency",
    x = "Recomputed Tokens",
    y = "Total Latency (s)"
  ) +
  base_theme

save_fig(p6, "fig6_recomputed_tokens_vs_latency.png", 8, 5)

# =========================
# 12. 导出表格数据
# =========================
write.csv(summary_df, file.path(fig_dir, "summary_metrics.csv"), row.names = FALSE)
write.csv(sample_df,  file.path(fig_dir, "sample_metrics.csv"), row.names = FALSE)

cat("Done. Figures saved to:", fig_dir, "\n")
