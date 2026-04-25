#!/usr/bin/env Rscript

# 说明：
# 1) 本脚本读取 outputs/useful/*.output
# 2) 自动解析 run_summary 与 sample_result
# 3) 生成多种论文可用图表到 graduation_project/analyse/figures
# 4) 图表标题使用英文，代码注释使用中文

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
repo_root <- getwd()
input_dir <- file.path(repo_root, "outputs", "useful")
out_dir <- file.path(repo_root, "graduation_project", "analyse")
fig_dir <- file.path(out_dir, "figures")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# 中文可读方法名与固定顺序
method_order <- c(
  "full_reuse", "full_prefill", "query_aware",
  "qaw_default", "qaw_no_suffix", "qaw_random_topk"
)

# 统一主题
base_theme <- theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

# 安全解析单行 JSON
safe_parse_json <- function(line) {
  line <- trimws(line)
  if (!startsWith(line, "{")) return(NULL)
  out <- tryCatch(fromJSON(line, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(out) || is.null(out$event)) return(NULL)
  out
}

# 推断数据集标签
infer_dataset <- function(dataset_path, file_name) {
  x <- tolower(paste(dataset_path %||% "", file_name, collapse = " "))
  if (grepl("musique", x)) return("musique")
  if (grepl("wikimqa", x)) return("wikimqa")
  if (grepl("samsum", x)) return("samsum")
  if (grepl("squad", x)) return("squad")
  if (grepl("blend", x)) return("blend_demo")
  "unknown"
}

`%||%` <- function(a, b) {
  if (is.null(a) || length(a) == 0) b else a
}

# 数值标签格式化：整数不带小数，浮点保留 3 位
format_value_label <- function(v) {
  ifelse(abs(v - round(v)) < 1e-9, sprintf("%d", as.integer(round(v))), sprintf("%.3f", v))
}

# 清理文件名显示：去掉时间前缀和 .output 后缀
clean_curve_file_label <- function(x) {
  y <- sub("^\\d{12}_", "", x)
  sub("\\.output$", "", y)
}

# 读取一个日志文件中的全部 JSON 事件
read_events_from_file <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  file_name <- basename(path)

  # 先找 run_start 用于补齐 dataset 信息
  run_start <- NULL
  events <- list()
  for (ln in lines) {
    ev <- safe_parse_json(ln)
    if (is.null(ev)) next
    if (is.null(run_start) && identical(ev$event, "run_start")) {
      run_start <- ev
    }
    events[[length(events) + 1]] <- ev
  }

  ds <- infer_dataset(run_start$dataset %||% "", file_name)
  script <- run_start$script %||% "unknown"

  lapply(events, function(ev) {
    ev$source_file <- file_name
    ev$dataset_name <- ds
    ev$script_name <- script
    ev
  })
}

output_files <- list.files(input_dir, pattern = "\\.output$", full.names = TRUE)
if (length(output_files) == 0) {
  stop(sprintf("No .output files found in %s", input_dir))
}

all_events <- unlist(lapply(output_files, read_events_from_file), recursive = FALSE)

run_summaries <- Filter(function(x) identical(x$event, "run_summary"), all_events)
sample_results <- Filter(function(x) identical(x$event, "sample_result"), all_events)

# 从 run_summary 中抽取 method 指标
extract_summary_metric <- function(summary_event, suffix_regex, metric_name) {
  keys <- names(summary_event)
  keys <- keys[grepl(suffix_regex, keys)]
  if (length(keys) == 0) return(NULL)

  # 将任意输入安全转成单个数值，异常时返回 NA
  to_numeric_scalar <- function(x) {
    v <- suppressWarnings(as.numeric(x))
    if (length(v) == 0) return(NA_real_)
    v[1]
  }

  rows <- list()
  for (k in keys) {
    v <- to_numeric_scalar(summary_event[[k]])
    if (is.na(v)) next
    method <- sub(suffix_regex, "", k)
    rows[[length(rows) + 1]] <- data.frame(
      source_file = summary_event$source_file,
      dataset = summary_event$dataset_name,
      script = summary_event$script_name,
      recomp_ratio = suppressWarnings(as.numeric(summary_event$recomp_ratio %||% NA)),
      method = method,
      metric = metric_name,
      value = v,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0) return(NULL)
  do.call(rbind, rows)
}

summary_total_df <- do.call(rbind, Filter(Negate(is.null), lapply(
  run_summaries,
  extract_summary_metric,
  suffix_regex = "_avg_total_s$",
  metric_name = "avg_total_s"
)))

summary_ttft_df <- do.call(rbind, Filter(Negate(is.null), lapply(
  run_summaries,
  extract_summary_metric,
  suffix_regex = "_avg_ttft_s$",
  metric_name = "avg_ttft_s"
)))

summary_f1_df <- do.call(rbind, Filter(Negate(is.null), lapply(
  run_summaries,
  extract_summary_metric,
  suffix_regex = "_avg_f1$",
  metric_name = "avg_f1"
)))

summary_rougel_df <- do.call(rbind, Filter(Negate(is.null), lapply(
  run_summaries,
  extract_summary_metric,
  suffix_regex = "_avg_rouge_l$",
  metric_name = "avg_rouge_l"
)))

summary_recompute_df <- do.call(rbind, Filter(Negate(is.null), lapply(
  run_summaries,
  extract_summary_metric,
  suffix_regex = "_true_recompute_count$",
  metric_name = "true_recompute_count"
)))

if (is.null(summary_total_df)) summary_total_df <- data.frame()
if (is.null(summary_ttft_df)) summary_ttft_df <- data.frame()
if (is.null(summary_f1_df)) summary_f1_df <- data.frame()
if (is.null(summary_rougel_df)) summary_rougel_df <- data.frame()
if (is.null(summary_recompute_df)) summary_recompute_df <- data.frame()

summary_quality_df <- rbind(summary_f1_df, summary_rougel_df)

# 从 sample_result 抽取方法级指标
extract_method_node <- function(ev, method, node) {
  if (is.null(node) || !is.list(node)) return(NULL)
  get_num <- function(x) suppressWarnings(as.numeric(x %||% NA))

  data.frame(
    source_file = ev$source_file,
    dataset = ev$dataset_name,
    script = ev$script_name,
    sample_idx = suppressWarnings(as.integer(ev$sample_idx %||% NA)),
    method = method,
    ttft_s = get_num(node$ttft_s),
    total_s = get_num(node$total_s),
    recomputed_tokens = get_num(node$recomputed_tokens),
    f1 = get_num(node$f1),
    rouge_l = get_num(node$rouge_l),
    true_recompute = as.logical(node$true_recompute %||% NA),
    stringsAsFactors = FALSE
  )
}

sample_rows <- list()
for (ev in sample_results) {
  # 常规四方法
  for (m in c("full_reuse", "full_prefill", "kv_diff", "query_aware")) {
    if (!is.null(ev[[m]])) {
      row <- extract_method_node(ev, m, ev[[m]])
      if (!is.null(row)) sample_rows[[length(sample_rows) + 1]] <- row
    }
  }
  # runner 风格 qaw_variants
  if (!is.null(ev$qaw_variants) && is.list(ev$qaw_variants)) {
    for (m in names(ev$qaw_variants)) {
      row <- extract_method_node(ev, m, ev$qaw_variants[[m]])
      if (!is.null(row)) sample_rows[[length(sample_rows) + 1]] <- row
    }
  }
}

sample_df <- if (length(sample_rows) > 0) do.call(rbind, sample_rows) else data.frame()

# 保存中间表，方便复用
write.csv(summary_total_df, file.path(out_dir, "summary_total_metrics.csv"), row.names = FALSE)
write.csv(summary_ttft_df, file.path(out_dir, "summary_ttft_metrics.csv"), row.names = FALSE)
write.csv(summary_quality_df, file.path(out_dir, "summary_quality_metrics.csv"), row.names = FALSE)
write.csv(summary_recompute_df, file.path(out_dir, "summary_recompute_counts.csv"), row.names = FALSE)
write.csv(sample_df, file.path(out_dir, "sample_method_metrics.csv"), row.names = FALSE)

save_plot <- function(plot_obj, file_name, width = 12, height = 7) {
  ggsave(
    filename = file.path(fig_dir, file_name),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = 220
  )
}

# 过滤非曲线日志
is_curve <- function(df) {
  !is.na(df$recomp_ratio)
}

# 图 1：平均 total 延迟（按 dataset+method）
if (nrow(summary_total_df) > 0) {
  df <- subset(summary_total_df, is.na(recomp_ratio) & method != "kv_diff")
  if (nrow(df) > 0) {
    agg <- aggregate(value ~ dataset + method, data = df, FUN = mean)
    agg$method <- factor(agg$method, levels = method_order)
    agg$label <- format_value_label(agg$value)
    p <- ggplot(agg, aes(x = method, y = value, fill = dataset)) +
      geom_col(position = position_dodge(width = 0.75), width = 0.68) +
      geom_text(
        aes(label = label),
        position = position_dodge(width = 0.75),
        vjust = -0.25,
        size = 3
      ) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
      labs(
        title = "Average Total Latency by Method and Dataset",
        x = "Method",
        y = "Average Total Latency (s)"
      ) +
      base_theme
    save_plot(p, "01_avg_total_latency_by_method_dataset.png")
  }
}

# 图 2：平均 TTFT（按 dataset+method）
if (nrow(summary_ttft_df) > 0) {
  df <- subset(summary_ttft_df, is.na(recomp_ratio) & method != "kv_diff")
  if (nrow(df) > 0) {
    agg <- aggregate(value ~ dataset + method, data = df, FUN = mean)
    agg$method <- factor(agg$method, levels = method_order)
    agg$label <- format_value_label(agg$value)
    p <- ggplot(agg, aes(x = method, y = value, fill = dataset)) +
      geom_col(position = position_dodge(width = 0.75), width = 0.68) +
      geom_text(
        aes(label = label),
        position = position_dodge(width = 0.75),
        vjust = -0.25,
        size = 3
      ) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
      labs(
        title = "Average TTFT by Method and Dataset",
        x = "Method",
        y = "Average TTFT (s)"
      ) +
      base_theme
    save_plot(p, "02_avg_ttft_by_method_dataset.png")
  }
}

# 图 3：质量指标（F1/ROUGE-L）
if (nrow(summary_quality_df) > 0) {
  df <- subset(summary_quality_df, is.na(recomp_ratio) & method != "kv_diff")
  if (nrow(df) > 0) {
    agg <- aggregate(value ~ dataset + method + metric, data = df, FUN = mean)
    agg$method <- factor(agg$method, levels = method_order)
    agg$label <- format_value_label(agg$value)
    p <- ggplot(agg, aes(x = method, y = value, fill = dataset)) +
      geom_col(position = position_dodge(width = 0.75), width = 0.68) +
      geom_text(
        aes(label = label),
        position = position_dodge(width = 0.75),
        vjust = -0.25,
        size = 2.7
      ) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
      facet_wrap(~metric, scales = "free_y") +
      labs(
        title = "Quality Metrics by Method and Dataset",
        x = "Method",
        y = "Metric Value"
      ) +
      base_theme
    save_plot(p, "03_quality_metrics_by_method_dataset.png")
  }
}

# 图 4：速度-质量散点图（run_summary 粒度）
if (nrow(summary_total_df) > 0 && nrow(summary_quality_df) > 0) {
  non_curve_total <- subset(summary_total_df, is.na(recomp_ratio))
  non_curve_quality <- subset(summary_quality_df, is.na(recomp_ratio))
  merged <- merge(
    non_curve_total,
    non_curve_quality,
    by = c("source_file", "dataset", "script", "method"),
    suffixes = c("_total", "_quality")
  )
  merged <- subset(merged, method != "kv_diff")
  if (nrow(merged) > 0) {
    p <- ggplot(merged, aes(x = value_total, y = value_quality, color = method, shape = dataset)) +
      geom_point(size = 3, alpha = 0.88) +
      facet_wrap(~metric_quality, scales = "free_y") +
      labs(
        title = "Speed-Quality Scatter (Run Summaries)",
        x = "Average Total Latency (s)",
        y = "Quality"
      ) +
      base_theme
    save_plot(p, "04_speed_quality_scatter_run_summary.png")
  }
}

# 图 5：曲线日志中，ratio 对 total latency 的影响
if (nrow(summary_total_df) > 0) {
  curve_total <- subset(summary_total_df, !is.na(recomp_ratio) & method %in% c("query_aware", "full_prefill", "full_reuse"))
  if (nrow(curve_total) > 0) {
    curve_total$method <- factor(curve_total$method, levels = method_order)
    curve_total$source_file_clean <- clean_curve_file_label(curve_total$source_file)
    p <- ggplot(curve_total, aes(x = recomp_ratio, y = value, color = method)) +
      geom_line(linewidth = 1.0) +
      geom_point(size = 2) +
      facet_wrap(~source_file_clean, scales = "free_y") +
      labs(
        title = "Latency Curves vs Recompute Ratio",
        x = "Recompute Ratio",
        y = "Average Total Latency (s)"
      ) +
      base_theme
    save_plot(p, "05_curve_latency_vs_ratio.png", width = 14, height = 8)
  }
}

# 图 6：曲线日志中，ratio 对质量的影响
if (nrow(summary_quality_df) > 0) {
  curve_quality <- subset(summary_quality_df, !is.na(recomp_ratio) & method %in% c("query_aware", "full_prefill", "full_reuse"))
  if (nrow(curve_quality) > 0) {
    curve_quality$method <- factor(curve_quality$method, levels = method_order)
    curve_quality$source_file_clean <- clean_curve_file_label(curve_quality$source_file)
    p <- ggplot(curve_quality, aes(x = recomp_ratio, y = value, color = method)) +
      geom_line(linewidth = 1.0) +
      geom_point(size = 2) +
      facet_grid(metric ~ source_file_clean, scales = "free_y") +
      labs(
        title = "Quality Curves vs Recompute Ratio",
        x = "Recompute Ratio",
        y = "Quality"
      ) +
      base_theme
    save_plot(p, "06_curve_quality_vs_ratio.png", width = 14, height = 8)
  }
}

# 图 7：曲线日志中的 speedup 与 quality delta（相对 full_prefill）
if (nrow(summary_total_df) > 0 && nrow(summary_quality_df) > 0) {
  ct <- subset(summary_total_df, !is.na(recomp_ratio) & method %in% c("query_aware", "full_prefill"))
  cq <- subset(summary_quality_df, !is.na(recomp_ratio) & method %in% c("query_aware", "full_prefill"))

  if (nrow(ct) > 0 && nrow(cq) > 0) {
    # total: reshape to wide
    key_cols <- c("source_file", "dataset", "script", "recomp_ratio")
    qa_t <- subset(ct, method == "query_aware")
    fp_t <- subset(ct, method == "full_prefill")
    t_merge <- merge(qa_t[, c(key_cols, "value")], fp_t[, c(key_cols, "value")],
                     by = key_cols, suffixes = c("_qa", "_fp"))

    qa_q <- subset(cq, method == "query_aware")
    fp_q <- subset(cq, method == "full_prefill")
    q_merge <- merge(qa_q[, c(key_cols, "metric", "value")], fp_q[, c(key_cols, "metric", "value")],
                     by = c(key_cols, "metric"), suffixes = c("_qa", "_fp"))

    if (nrow(t_merge) > 0) {
      t_merge$speedup <- t_merge$value_fp / t_merge$value_qa
      q_delta <- data.frame()
      if (nrow(q_merge) > 0) {
        q_merge$quality_delta <- q_merge$value_qa - q_merge$value_fp
        q_delta <- q_merge[, c(key_cols, "metric", "quality_delta")]
      }

      sp <- t_merge[, c(key_cols, "speedup")]
      sp$metric <- "speedup_vs_full_prefill"
      names(sp)[names(sp) == "speedup"] <- "value"

      if (nrow(q_delta) > 0) {
        qd <- q_delta
        qd$metric <- paste0("quality_delta_", qd$metric)
        names(qd)[names(qd) == "quality_delta"] <- "value"
        trend <- rbind(sp, qd[, c(key_cols, "metric", "value")])
      } else {
        trend <- sp
      }
      trend$source_file_clean <- clean_curve_file_label(trend$source_file)

      p <- ggplot(trend, aes(x = recomp_ratio, y = value, color = source_file_clean)) +
        geom_line(linewidth = 1.0) +
        geom_point(size = 1.8) +
        facet_wrap(~metric, scales = "free_y") +
        labs(
          title = "Speedup and Quality Delta vs Recompute Ratio",
          x = "Recompute Ratio",
          y = "Value"
        ) +
        base_theme
      save_plot(p, "07_curve_speedup_quality_delta.png", width = 14, height = 8)
    }
  }
}

# 图 8：sample 粒度重算 token 分布
if (nrow(sample_df) > 0) {
  df <- subset(sample_df, method != "kv_diff" & !is.na(recomputed_tokens) & recomputed_tokens > 0)
  if (nrow(df) > 0) {
    df$method <- factor(df$method, levels = method_order)
    p <- ggplot(df, aes(x = method, y = recomputed_tokens, fill = method)) +
      geom_boxplot(outlier.alpha = 0.25) +
      facet_wrap(~dataset, scales = "free_y") +
      labs(
        title = "Recomputed Tokens Distribution (Sample Level)",
        x = "Method",
        y = "Recomputed Tokens"
      ) +
      base_theme +
      guides(fill = "none")
    save_plot(p, "08_sample_recomputed_tokens_boxplot.png", width = 14, height = 8)
  }
}

# 图 9：sample 粒度 total latency 分布
if (nrow(sample_df) > 0) {
  df <- subset(sample_df, method != "kv_diff" & !is.na(total_s))
  if (nrow(df) > 0) {
    df$method <- factor(df$method, levels = method_order)
    p <- ggplot(df, aes(x = method, y = total_s, fill = method)) +
      geom_boxplot(outlier.alpha = 0.2) +
      facet_wrap(~dataset, scales = "free_y") +
      labs(
        title = "Sample-level Total Latency Distribution",
        x = "Method",
        y = "Total Latency (s)"
      ) +
      base_theme +
      guides(fill = "none")
    save_plot(p, "09_sample_total_latency_boxplot.png", width = 14, height = 8)
  }
}

# 图 10：相对 full_prefill 的胜率（sample 粒度）
if (nrow(sample_df) > 0) {
  # 先按 source_file + sample_idx 找 full_prefill 基线
  base_rows <- subset(sample_df, method == "full_prefill" & !is.na(total_s))
  if (nrow(base_rows) > 0) {
    merged <- merge(
      sample_df,
      base_rows[, c("source_file", "sample_idx", "total_s", "f1", "rouge_l")],
      by = c("source_file", "sample_idx"),
      suffixes = c("", "_fp")
    )
    merged <- subset(merged, method != "full_prefill" & !is.na(total_s_fp))
    merged <- subset(merged, method != "kv_diff")

    if (nrow(merged) > 0) {
      merged$faster_than_fp <- merged$total_s < merged$total_s_fp

      # 自动选择质量字段：优先 f1，否则 rouge_l
      merged$quality <- ifelse(!is.na(merged$f1), merged$f1, merged$rouge_l)
      merged$quality_fp <- ifelse(!is.na(merged$f1_fp), merged$f1_fp, merged$rouge_l_fp)
      merged$quality_not_lower <- ifelse(
        is.na(merged$quality) | is.na(merged$quality_fp),
        NA,
        merged$quality >= merged$quality_fp
      )

      agg_fast <- aggregate(faster_than_fp ~ dataset + method, data = merged, FUN = mean)
      agg_fast$metric <- "faster_than_full_prefill_rate"
      names(agg_fast)[names(agg_fast) == "faster_than_fp"] <- "value"

      valid_quality <- subset(merged, !is.na(quality_not_lower))
      agg_q <- data.frame()
      if (nrow(valid_quality) > 0) {
        agg_q <- aggregate(quality_not_lower ~ dataset + method, data = valid_quality, FUN = mean)
        agg_q$metric <- "quality_not_lower_rate"
        names(agg_q)[names(agg_q) == "quality_not_lower"] <- "value"
      }

      win_df <- rbind(agg_fast, agg_q)
      win_df$method <- factor(win_df$method, levels = method_order)
      win_df$label <- format_value_label(win_df$value)

      p <- ggplot(win_df, aes(x = method, y = value, fill = dataset)) +
        geom_col(position = position_dodge(width = 0.75), width = 0.68) +
        geom_text(
          aes(label = label),
          position = position_dodge(width = 0.75),
          vjust = -0.25,
          size = 2.8
        ) +
        scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
        facet_wrap(~metric, scales = "free_y") +
        labs(
          title = "Method Win Rate vs Full Prefill (Sample Level)",
          x = "Method",
          y = "Rate"
        ) +
        base_theme
      save_plot(p, "10_method_win_rate_vs_full_prefill.png", width = 14, height = 8)
    }
  }
}

# 图 11：true recompute 次数（run_summary 粒度）
if (nrow(summary_recompute_df) > 0) {
  df <- subset(summary_recompute_df, method != "kv_diff")
  df$method <- factor(df$method, levels = method_order)
  df$label <- format_value_label(df$value)
  p <- ggplot(df, aes(x = method, y = value, fill = dataset)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    geom_text(
      aes(label = label),
      position = position_dodge(width = 0.75),
      vjust = -0.25,
      size = 3
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(
      title = "True Recompute Counts in Run Summaries",
      x = "Method",
      y = "Count"
    ) +
    base_theme
  save_plot(p, "11_true_recompute_counts.png")
}

cat(sprintf("Done. Parsed files: %d\n", length(output_files)))
cat(sprintf("Figures saved to: %s\n", fig_dir))
