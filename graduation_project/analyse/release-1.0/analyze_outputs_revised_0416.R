#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
})

repo_root <- getwd()
input_dirs <- c(
  file.path(repo_root, "outputs", "useful"),
  file.path(repo_root, "outputs", "0416_sum")
)
out_dir <- file.path(repo_root, "graduation_project", "analyse", "reanalysis_0416")
fig_dir <- file.path(out_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

`%||%` <- function(a, b) {
  if (is.null(a) || length(a) == 0) b else a
}

safe_num <- function(x) {
  y <- suppressWarnings(as.numeric(x))
  if (length(y) == 0) return(NA_real_)
  y[1]
}

safe_int <- function(x) {
  y <- suppressWarnings(as.integer(x))
  if (length(y) == 0) return(NA_integer_)
  y[1]
}

safe_parse_json <- function(line) {
  line <- trimws(line)
  if (!startsWith(line, "{")) return(NULL)
  out <- tryCatch(fromJSON(line, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(out) || is.null(out$event)) return(NULL)
  out
}

fmt_key_num <- function(x) {
  if (is.null(x) || length(x) == 0) return(NA_character_)
  x <- suppressWarnings(as.numeric(x))
  if (length(x) == 0 || is.na(x[1])) return(NA_character_)
  format(x[1], scientific = FALSE, trim = TRUE, digits = 15)
}

build_baseline_group_key <- function(dataset_name, count_value, max_new_tokens,
                                     temperature, top_p, model_name) {
  if (is.null(dataset_name) || is.na(dataset_name) || dataset_name == "") return(NA_character_)
  count_part <- fmt_key_num(count_value)
  max_new_tokens_part <- fmt_key_num(max_new_tokens)
  temperature_part <- fmt_key_num(temperature)
  top_p_part <- fmt_key_num(top_p)
  model_part <- as.character(model_name %||% NA_character_)
  if (any(is.na(c(count_part, max_new_tokens_part, temperature_part, top_p_part, model_part)))) {
    return(NA_character_)
  }
  paste(
    dataset_name,
    paste0("count=", count_part),
    paste0("max_new_tokens=", max_new_tokens_part),
    paste0("temperature=", temperature_part),
    paste0("top_p=", top_p_part),
    paste0("model=", model_part),
    sep = "|"
  )
}

infer_dataset <- function(file_name, dataset_field = NULL) {
  x <- tolower(paste(file_name, dataset_field %||% "", collapse = " "))
  if (str_detect(x, "musique")) return("musique")
  if (str_detect(x, "wikimqa")) return("wikimqa")
  if (str_detect(x, "samsum")) return("samsum")
  if (str_detect(x, "squad")) return("squad")
  if (str_detect(x, "blend")) return("blend_demo")
  return("unknown")
}

clean_file_label <- function(x) {
  x %>%
    basename() %>%
    str_remove("^\\d{12}_") %>%
    str_remove("\\.output$")
}

detect_curve_mode <- function(script_name, run_start) {
  if (!is.null(run_start$curve_mode)) {
    return(as.character(run_start$curve_mode))
  }
  x <- tolower(script_name %||% "")
  if (str_detect(x, "suffix_curve")) return("suffix_only")
  if (str_detect(x, "curve")) return("ratio_only")
  return("non_curve")
}

is_experiment_curve <- function(script_name, run_start) {
  mode <- detect_curve_mode(script_name, run_start)
  mode %in% c("ratio_only", "suffix_only")
}

metric_pretty <- c(
  avg_ttft_s = "Average TTFT (s)",
  avg_total_s = "Average Total Latency (s)",
  avg_f1 = "Average F1",
  avg_rouge_l = "Average ROUGE-L",
  speedup_vs_prefill = "Speedup vs Full Prefill",
  f1_retention = "F1 Retention"
)

method_colors <- c(
  full_prefill = "#D55E00",
  full_reuse = "#0072B2",
  query_aware = "#009E73",
  kv_diff = "#7F7F7F",
  qaw_default = "#009E73",
  qaw_no_suffix = "#E69F00",
  qaw_random_topk = "#CC79A7"
)

base_theme <- theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 20, hjust = 1),
    legend.position = "right",
    strip.background = element_rect(fill = "#F2F2F2"),
    panel.grid.minor = element_blank()
  )

save_plot <- function(plot_obj, file_name, width = 10, height = 6) {
  ggsave(
    filename = file.path(fig_dir, file_name),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = 260
  )
}

figure_registry <- tibble(
  file_name = character(),
  title = character(),
  chart_type = character(),
  data_scope = character(),
  meaning = character(),
  value = character(),
  recommend_for_paper = logical(),
  paper_order = integer()
)

register_figure <- function(file_name, title, chart_type, data_scope,
                            meaning, value, recommend_for_paper = FALSE,
                            paper_order = NA_integer_) {
  figure_registry <<- bind_rows(
    figure_registry,
    tibble(
      file_name = file_name,
      title = title,
      chart_type = chart_type,
      data_scope = data_scope,
      meaning = meaning,
      value = value,
      recommend_for_paper = recommend_for_paper,
      paper_order = paper_order
    )
  )
}

extract_summary_rows <- function(ev, meta_row, summary_order, drop_curve_first) {
  suffix_map <- list(
    avg_ttft_s = "_avg_ttft_s$",
    avg_total_s = "_avg_total_s$",
    avg_f1 = "_avg_f1$",
    avg_rouge_l = "_avg_rouge_l$",
    true_recompute_count = "_true_recompute_count$"
  )
  rows <- list()
  for (metric_name in names(suffix_map)) {
    hit_keys <- names(ev)[str_detect(names(ev), suffix_map[[metric_name]])]
    if (length(hit_keys) == 0) next
    for (k in hit_keys) {
      method <- str_remove(k, suffix_map[[metric_name]])
      rows[[length(rows) + 1]] <- tibble(
        source_file = meta_row$source_file,
        source_path = meta_row$source_path,
        source_bucket = meta_row$source_bucket,
        script = meta_row$script,
        dataset = meta_row$dataset,
        file_label = meta_row$file_label,
        is_curve = meta_row$is_curve,
        curve_mode = meta_row$curve_mode,
        baseline_only = meta_row$baseline_only,
        baseline_mode = meta_row$baseline_mode,
        baseline_group_key = meta_row$baseline_group_key,
        summary_order = summary_order,
        drop_curve_first = drop_curve_first,
        sample_count = safe_int(ev$sample_count %||% meta_row$count_hint),
        recomp_ratio = safe_num(ev$recomp_ratio),
        suffix_len = safe_num(ev$suffix_len),
        metric = metric_name,
        method = method,
        value = safe_num(ev[[k]]),
        qaw_ratio_min = meta_row$qaw_ratio_min,
        qaw_ratio_max = meta_row$qaw_ratio_max,
        qaw_ratio_step = meta_row$qaw_ratio_step,
        qaw_ratio_fixed = meta_row$qaw_ratio_fixed,
        suffix_len_min = meta_row$suffix_len_min,
        suffix_len_max = meta_row$suffix_len_max,
        suffix_len_step = meta_row$suffix_len_step
      )
    }
  }
  bind_rows(rows)
}

extract_method_node <- function(ev, meta_row, method, node) {
  if (is.null(node) || !is.list(node)) return(NULL)
  tibble(
    source_file = meta_row$source_file,
    source_path = meta_row$source_path,
    source_bucket = meta_row$source_bucket,
    script = meta_row$script,
    dataset = meta_row$dataset,
    file_label = meta_row$file_label,
    is_curve = meta_row$is_curve,
    baseline_only = meta_row$baseline_only,
    baseline_mode = meta_row$baseline_mode,
    baseline_group_key = meta_row$baseline_group_key,
    sample_idx = safe_int(ev$sample_idx),
    method = method,
    ttft_s = safe_num(node$ttft_s),
    total_s = safe_num(node$total_s),
    recomputed_tokens = safe_num(node$recomputed_tokens),
    f1 = safe_num(node$f1),
    rouge_l = safe_num(node$rouge_l),
    true_recompute = as.logical(node$true_recompute %||% NA)
  )
}

all_files <- map(input_dirs, ~ list.files(.x, pattern = "\\.output$", full.names = TRUE)) %>%
  unlist() %>%
  sort()

file_catalog <- list()
ignored_files <- list()
summary_rows <- list()
sample_rows <- list()

for (path in all_files) {
  bucket <- basename(dirname(path))
  file_name <- basename(path)
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  events <- map(lines, safe_parse_json) %>% compact()

  if (length(events) == 0) {
    ignored_files[[length(ignored_files) + 1]] <- tibble(
      source_file = file_name,
      source_path = path,
      reason = "no_json_events"
    )
    next
  }

  run_start <- keep(events, ~ identical(.x$event, "run_start"))
  run_start <- if (length(run_start) > 0) run_start[[1]] else NULL
  script_name <- as.character(run_start$script %||% "unknown")
  dataset_name <- as.character(run_start$dataset_name %||%
                                 infer_dataset(file_name, run_start$dataset %||% ""))
  curve_mode <- detect_curve_mode(script_name, run_start)
  is_curve <- is_experiment_curve(script_name, run_start)

  if (basename(script_name) == "0416_sum.py" || identical(script_name, "example/0416_sum.py")) {
    ignored_files[[length(ignored_files) + 1]] <- tibble(
      source_file = file_name,
      source_path = path,
      reason = "orchestration_log"
    )
    next
  }

  run_summaries <- keep(events, function(ev) {
    identical(ev$event, "run_summary") &&
      any(str_detect(names(ev), "_avg_ttft_s$|_avg_total_s$|_avg_f1$|_avg_rouge_l$"))
  })

  sample_results <- keep(events, ~ identical(.x$event, "sample_result"))

  if (length(run_summaries) == 0 && length(sample_results) == 0) {
    ignored_files[[length(ignored_files) + 1]] <- tibble(
      source_file = file_name,
      source_path = path,
      reason = "no_experiment_metrics"
    )
    next
  }

  count_hint <- safe_int(run_start$count %||% NA)
  if (is.na(count_hint) && length(run_summaries) > 0) {
    count_hint <- safe_int(run_summaries[[1]]$sample_count)
  }
  baseline_only <- isTRUE(run_start$baseline_only %||% FALSE)
  baseline_mode <- as.character(run_start$baseline_mode %||%
                                  ifelse(baseline_only, "standalone_hot", "native"))
  max_new_tokens <- safe_int(run_start$max_new_tokens)
  temperature <- safe_num(run_start$temperature)
  top_p <- safe_num(run_start$top_p)
  model_name <- as.character(run_start$model %||% NA_character_)
  baseline_group_key <- as.character(
    run_start$baseline_group_key %||%
      build_baseline_group_key(
        dataset_name = dataset_name,
        count_value = count_hint,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_p = top_p,
        model_name = model_name
      )
  )

  meta_row <- tibble(
    source_file = file_name,
    source_path = path,
    source_bucket = bucket,
    file_label = clean_file_label(file_name),
    script = script_name,
    dataset = dataset_name,
    is_curve = is_curve,
    curve_mode = curve_mode,
    count_hint = count_hint,
    baseline_only = baseline_only,
    baseline_mode = baseline_mode,
    baseline_group_key = baseline_group_key,
    max_new_tokens = max_new_tokens,
    temperature = temperature,
    top_p = top_p,
    model = model_name,
    qaw_ratio_min = safe_num(run_start$qaw_ratio_min),
    qaw_ratio_max = safe_num(run_start$qaw_ratio_max),
    qaw_ratio_step = safe_num(run_start$qaw_ratio_step),
    qaw_ratio_fixed = safe_num(run_start$qaw_ratio),
    suffix_len_min = safe_num(run_start$suffix_len_min),
    suffix_len_max = safe_num(run_start$suffix_len_max),
    suffix_len_step = safe_num(run_start$suffix_len_step),
    suffix_len_fixed = safe_num(run_start$suffix_len)
  )

  file_catalog[[length(file_catalog) + 1]] <- meta_row %>%
    mutate(
      run_summary_count = length(run_summaries),
      sample_result_count = length(sample_results),
      curve_first_group_dropped = is_curve && length(run_summaries) >= 1
    )

  if (length(run_summaries) > 0) {
    for (i in seq_along(run_summaries)) {
      summary_rows[[length(summary_rows) + 1]] <- extract_summary_rows(
        run_summaries[[i]],
        meta_row = meta_row,
        summary_order = i,
        drop_curve_first = is_curve && i == 1
      )
    }
  }

  if (length(sample_results) > 0) {
    for (ev in sample_results) {
      for (m in c("full_reuse", "full_prefill", "kv_diff", "query_aware")) {
        if (!is.null(ev[[m]])) {
          row <- extract_method_node(ev, meta_row, m, ev[[m]])
          if (!is.null(row)) sample_rows[[length(sample_rows) + 1]] <- row
        }
      }
      if (!is.null(ev$qaw_variants) && is.list(ev$qaw_variants)) {
        for (m in names(ev$qaw_variants)) {
          row <- extract_method_node(ev, meta_row, m, ev$qaw_variants[[m]])
          if (!is.null(row)) sample_rows[[length(sample_rows) + 1]] <- row
        }
      }
    }
  }
}

file_catalog_df <- bind_rows(file_catalog)
ignored_files_df <- bind_rows(ignored_files)
summary_long_raw <- bind_rows(summary_rows)
sample_df_raw <- bind_rows(sample_rows)

if (nrow(summary_long_raw) == 0) {
  stop("未解析到任何可用的实验指标。")
}

summary_long_clean <- summary_long_raw %>%
  mutate(
    method_group = case_when(
      method %in% c("query_aware", "qaw_default") ~ "query_aware",
      method == "qaw_no_suffix" ~ "qaw_no_suffix",
      method == "qaw_random_topk" ~ "qaw_random_topk",
      TRUE ~ method
    )
  )

curve_dropped_df <- summary_long_raw %>%
  filter(drop_curve_first)

sample_df_raw <- sample_df_raw %>%
  mutate(
    method_group = case_when(
      method %in% c("query_aware", "qaw_default") ~ "query_aware",
      method == "qaw_no_suffix" ~ "qaw_no_suffix",
      method == "qaw_random_topk" ~ "qaw_random_topk",
      TRUE ~ method
    )
  )

summary_baseline_override <- summary_long_clean %>%
  filter(baseline_only) %>%
  filter(!drop_curve_first) %>%
  filter(!is.na(value)) %>%
  filter(method_group %in% c("full_prefill", "full_reuse")) %>%
  arrange(source_file, summary_order) %>%
  group_by(baseline_group_key, metric, method_group) %>%
  summarise(
    override_value = dplyr::last(value),
    baseline_source_file = dplyr::last(source_file),
    .groups = "drop"
  )

summary_long_clean <- summary_long_clean %>%
  filter(!baseline_only) %>%
  filter(!drop_curve_first) %>%
  filter(!is.na(value)) %>%
  left_join(
    summary_baseline_override,
    by = c("baseline_group_key", "metric", "method_group")
  ) %>%
  mutate(
    metric_source = case_when(
      method_group %in% c("full_prefill", "full_reuse") & !is.na(override_value) ~
        "standalone_baseline_override",
      TRUE ~ "native"
    ),
    baseline_source_file = if_else(
      metric_source == "standalone_baseline_override",
      baseline_source_file,
      NA_character_
    ),
    value = if_else(
      metric_source == "standalone_baseline_override",
      override_value,
      value
    )
  ) %>%
  select(-override_value)

sample_baseline_override <- sample_df_raw %>%
  filter(baseline_only) %>%
  filter(!is_curve) %>%
  filter(!is.na(sample_idx)) %>%
  filter(method_group %in% c("full_prefill", "full_reuse")) %>%
  arrange(source_file, sample_idx) %>%
  group_by(baseline_group_key, sample_idx, method_group) %>%
  summarise(
    override_ttft_s = dplyr::last(ttft_s),
    override_total_s = dplyr::last(total_s),
    override_recomputed_tokens = dplyr::last(recomputed_tokens),
    override_f1 = dplyr::last(f1),
    override_rouge_l = dplyr::last(rouge_l),
    override_true_recompute = dplyr::last(true_recompute),
    baseline_source_file = dplyr::last(source_file),
    .groups = "drop"
  )

sample_df_clean <- sample_df_raw %>%
  filter(!baseline_only) %>%
  filter(!is_curve) %>%
  filter(!is.na(sample_idx)) %>%
  left_join(
    sample_baseline_override,
    by = c("baseline_group_key", "sample_idx", "method_group")
  ) %>%
  mutate(
    metric_source = case_when(
      method_group %in% c("full_prefill", "full_reuse") & !is.na(baseline_source_file) ~
        "standalone_baseline_override",
      TRUE ~ "native"
    ),
    ttft_s = if_else(metric_source == "standalone_baseline_override",
                     coalesce(override_ttft_s, ttft_s), ttft_s),
    total_s = if_else(metric_source == "standalone_baseline_override",
                      coalesce(override_total_s, total_s), total_s),
    recomputed_tokens = if_else(metric_source == "standalone_baseline_override",
                                coalesce(override_recomputed_tokens, recomputed_tokens),
                                recomputed_tokens),
    f1 = if_else(metric_source == "standalone_baseline_override",
                 coalesce(override_f1, f1), f1),
    rouge_l = if_else(metric_source == "standalone_baseline_override",
                      coalesce(override_rouge_l, rouge_l), rouge_l),
    true_recompute = if_else(metric_source == "standalone_baseline_override",
                             coalesce(override_true_recompute, true_recompute),
                             true_recompute)
  ) %>%
  select(-starts_with("override_"))

write.csv(file_catalog_df, file.path(out_dir, "file_catalog.csv"), row.names = FALSE)
write.csv(ignored_files_df, file.path(out_dir, "ignored_files.csv"), row.names = FALSE)
write.csv(summary_long_raw, file.path(out_dir, "summary_metrics_raw.csv"), row.names = FALSE)
write.csv(summary_long_clean, file.path(out_dir, "summary_metrics_clean.csv"), row.names = FALSE)
write.csv(curve_dropped_df, file.path(out_dir, "curve_first_group_dropped.csv"), row.names = FALSE)
write.csv(sample_df_clean, file.path(out_dir, "sample_metrics_clean.csv"), row.names = FALSE)

curve_file_comparability <- file_catalog_df %>%
  filter(is_curve) %>%
  mutate(
    count_for_compare = count_hint,
    compare_group = case_when(
      curve_mode == "ratio_only" ~ paste(
        "ratio",
        count_for_compare,
        qaw_ratio_min,
        qaw_ratio_max,
        qaw_ratio_step,
        suffix_len_fixed,
        sep = "|"
      ),
      curve_mode == "suffix_only" ~ paste(
        "suffix",
        count_for_compare,
        qaw_ratio_fixed,
        suffix_len_min,
        suffix_len_max,
        suffix_len_step,
        sep = "|"
      ),
      TRUE ~ NA_character_
    )
  )

comparable_groups_df <- curve_file_comparability %>%
  filter(!is.na(compare_group)) %>%
  group_by(compare_group, curve_mode, count_for_compare) %>%
  summarise(
    dataset_count = n_distinct(dataset),
    datasets = paste(sort(unique(dataset)), collapse = ", "),
    files = paste(sort(unique(file_label)), collapse = " | "),
    qaw_ratio_min = first(qaw_ratio_min),
    qaw_ratio_max = first(qaw_ratio_max),
    qaw_ratio_step = first(qaw_ratio_step),
    qaw_ratio_fixed = first(qaw_ratio_fixed),
    suffix_len_min = first(suffix_len_min),
    suffix_len_max = first(suffix_len_max),
    suffix_len_step = first(suffix_len_step),
    suffix_len_fixed = first(suffix_len_fixed),
    .groups = "drop"
  ) %>%
  arrange(curve_mode, compare_group)

write.csv(comparable_groups_df, file.path(out_dir, "comparable_curve_groups.csv"), row.names = FALSE)

curve_clean <- summary_long_clean %>%
  filter(is_curve) %>%
  mutate(
    curve_x = case_when(
      curve_mode == "suffix_only" ~ suffix_len,
      TRUE ~ recomp_ratio
    ),
    curve_x_name = case_when(
      curve_mode == "suffix_only" ~ "suffix_len",
      TRUE ~ "recomp_ratio"
    )
  )

noncurve_clean <- summary_long_clean %>%
  filter(!is_curve)

# 图 1/2：单数据集主实验方法对比
main_metrics <- c("avg_ttft_s", "avg_total_s", "avg_f1")
main_plot_df <- noncurve_clean %>%
  filter(metric %in% main_metrics) %>%
  filter(method_group %in% c("full_prefill", "full_reuse", "query_aware")) %>%
  mutate(metric_label = metric_pretty[metric])

for (ds in sort(unique(main_plot_df$dataset))) {
  ds_df <- main_plot_df %>%
    filter(dataset == ds) %>%
    group_by(dataset, method_group, metric_label) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop")
  if (nrow(ds_df) == 0) next
  p <- ggplot(ds_df, aes(x = method_group, y = value, fill = method_group)) +
    geom_col(width = 0.68) +
    geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.3, size = 3.4) +
    facet_wrap(~metric_label, scales = "free_y", nrow = 1) +
    scale_fill_manual(values = method_colors, guide = "none") +
    labs(
      title = sprintf("Main Method Comparison on %s", str_to_title(ds)),
      x = "Method",
      y = NULL
    ) +
    base_theme
  file_name <- sprintf("fig_main_methods_%s.png", ds)
  save_plot(p, file_name, width = 12, height = 4.8)
  register_figure(
    file_name = file_name,
    title = sprintf("%s 主实验方法对比", ds),
    chart_type = "Grouped Bar",
    data_scope = sprintf("%s 非曲线主实验", ds),
    meaning = "比较 full_prefill、full_reuse 和 query_aware 在 TTFT、总时延和 F1 上的整体表现。",
    value = "适合做该数据集主结果概览，帮助快速说明 query_aware 的总体位置。",
    recommend_for_paper = ds %in% c("musique", "wikimqa"),
    paper_order = case_when(
      ds == "musique" ~ 1L,
      ds == "wikimqa" ~ 2L,
      TRUE ~ NA_integer_
    )
  )
}

# 比例曲线绘图函数
plot_ratio_curve_file <- function(file_label_target) {
  df <- curve_clean %>%
    filter(file_label == file_label_target, curve_mode == "ratio_only") %>%
    filter(metric %in% c("avg_ttft_s", "avg_total_s", "avg_f1")) %>%
    filter(method_group %in% c("query_aware", "full_prefill", "full_reuse")) %>%
    mutate(metric_label = metric_pretty[metric])
  if (nrow(df) == 0) return(NULL)
  p <- ggplot(df, aes(x = curve_x, y = value, color = method_group)) +
    geom_line(linewidth = 1.0) +
    geom_point(size = 2) +
    facet_wrap(~metric_label, scales = "free_y", ncol = 1) +
    scale_color_manual(values = method_colors) +
    labs(
      title = sprintf("Ratio Sweep on %s", file_label_target),
      x = "Recompute Ratio",
      y = NULL,
      color = "Method"
    ) +
    base_theme
  file_name <- sprintf("fig_ratio_curve_%s.png", file_label_target)
  save_plot(p, file_name, width = 9, height = 10)
  register_figure(
    file_name = file_name,
    title = sprintf("%s 比例扫描曲线", file_label_target),
    chart_type = "Line",
    data_scope = file_label_target,
    meaning = "展示重算比例变化对 query_aware 延迟与质量的影响，并与 full_prefill/full_reuse 基线对照。",
    value = "最适合解释参数敏感性和效率-质量权衡，是论文中参数分析的核心图。",
    recommend_for_paper = file_label_target %in% c("0416_musique_curve_025_100", "wikimqa_curve"),
    paper_order = case_when(
      file_label_target == "0416_musique_curve_025_100" ~ 3L,
      file_label_target == "wikimqa_curve" ~ 4L,
      TRUE ~ NA_integer_
    )
  )
}

ratio_file_labels <- curve_clean %>%
  filter(curve_mode == "ratio_only") %>%
  pull(file_label) %>%
  unique() %>%
  sort()
walk(ratio_file_labels, plot_ratio_curve_file)

# 数据集内 ratio 热图
for (ds in sort(unique(curve_clean$dataset))) {
  ds_total <- curve_clean %>%
    filter(dataset == ds, curve_mode == "ratio_only", metric == "avg_total_s",
           method_group == "query_aware") %>%
    mutate(file_label = factor(file_label, levels = unique(file_label)))
  if (nrow(ds_total) > 1) {
    p <- ggplot(ds_total, aes(x = curve_x, y = file_label, fill = value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = sprintf("%.3f", value)), size = 3) +
      scale_fill_gradient(low = "#DEEBF7", high = "#08519C") +
      labs(
        title = sprintf("QAW Total Latency Heatmap on %s", str_to_title(ds)),
        x = "Recompute Ratio",
        y = "Curve File",
        fill = "Total (s)"
      ) +
      base_theme
    file_name <- sprintf("fig_ratio_heatmap_total_%s.png", ds)
    save_plot(p, file_name, width = 10, height = 5)
    register_figure(
      file_name = file_name,
      title = sprintf("%s 总时延热图", ds),
      chart_type = "Heatmap",
      data_scope = sprintf("%s 所有 ratio 曲线文件", ds),
      meaning = "比较同一数据集下不同 ratio 曲线文件在各比例点上的 qaw 总时延分布。",
      value = "适合快速筛选低时延区间，也能看出不同曲线配置间的一致性。",
      recommend_for_paper = FALSE
    )
  }

  ds_f1 <- curve_clean %>%
    filter(dataset == ds, curve_mode == "ratio_only", metric == "avg_f1",
           method_group == "query_aware") %>%
    mutate(file_label = factor(file_label, levels = unique(file_label)))
  if (nrow(ds_f1) > 1) {
    p <- ggplot(ds_f1, aes(x = curve_x, y = file_label, fill = value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = sprintf("%.3f", value)), size = 3) +
      scale_fill_gradient(low = "#FEE0D2", high = "#A50F15") +
      labs(
        title = sprintf("QAW F1 Heatmap on %s", str_to_title(ds)),
        x = "Recompute Ratio",
        y = "Curve File",
        fill = "F1"
      ) +
      base_theme
    file_name <- sprintf("fig_ratio_heatmap_f1_%s.png", ds)
    save_plot(p, file_name, width = 10, height = 5)
    register_figure(
      file_name = file_name,
      title = sprintf("%s F1 热图", ds),
      chart_type = "Heatmap",
      data_scope = sprintf("%s 所有 ratio 曲线文件", ds),
      meaning = "比较同一数据集下不同 ratio 曲线文件在各比例点上的 qaw 质量分布。",
      value = "适合寻找相对稳定的高质量区间，与时延热图结合可做参数筛选。",
      recommend_for_paper = FALSE
    )
  }
}

# 数据集内 speed-quality tradeoff
trade_df <- curve_clean %>%
  filter(curve_mode == "ratio_only",
         metric %in% c("avg_total_s", "avg_f1"),
         method_group %in% c("query_aware", "full_prefill")) %>%
  select(dataset, file_label, curve_x, method_group, metric, value) %>%
  pivot_wider(names_from = metric, values_from = value)

trade_base <- trade_df %>%
  filter(method_group == "full_prefill") %>%
  group_by(dataset, file_label) %>%
  summarise(
    base_total = mean(avg_total_s, na.rm = TRUE),
    base_f1 = mean(avg_f1, na.rm = TRUE),
    .groups = "drop"
  )

trade_qaw <- trade_df %>%
  filter(method_group == "query_aware") %>%
  left_join(trade_base, by = c("dataset", "file_label")) %>%
  mutate(
    speedup_vs_prefill = base_total / avg_total_s,
    f1_retention = avg_f1 / base_f1
  )

for (ds in sort(unique(trade_qaw$dataset))) {
  ds_df <- trade_qaw %>% filter(dataset == ds)
  if (nrow(ds_df) == 0) next
  p <- ggplot(ds_df, aes(x = speedup_vs_prefill, y = f1_retention, color = file_label)) +
    geom_path(linewidth = 0.9) +
    geom_point(aes(size = curve_x), alpha = 0.9) +
    labs(
      title = sprintf("Speed-Quality Tradeoff on %s", str_to_title(ds)),
      x = "Speedup vs Full Prefill",
      y = "F1 Retention",
      color = "Curve File",
      size = "Ratio"
    ) +
    base_theme
  file_name <- sprintf("fig_tradeoff_%s.png", ds)
  save_plot(p, file_name, width = 9, height = 6)
  register_figure(
    file_name = file_name,
    title = sprintf("%s 速度-质量权衡图", ds),
    chart_type = "Scatter Path",
    data_scope = sprintf("%s ratio 曲线", ds),
    meaning = "用速度提升和质量保持率同时观察不同比例点的综合收益。",
    value = "适合论文中解释“最优点不是单看延迟或单看F1，而是看两者平衡”。",
    recommend_for_paper = ds == "musique",
    paper_order = ifelse(ds == "musique", 5L, NA_integer_)
  )
}

# suffix 曲线单文件图
suffix_file_labels <- curve_clean %>%
  filter(curve_mode == "suffix_only") %>%
  pull(file_label) %>%
  unique() %>%
  sort()

for (label in suffix_file_labels) {
  df <- curve_clean %>%
    filter(file_label == label, curve_mode == "suffix_only",
           metric %in% c("avg_ttft_s", "avg_total_s", "avg_f1"),
           method_group %in% c("query_aware", "full_prefill", "full_reuse")) %>%
    mutate(metric_label = metric_pretty[metric])
  if (nrow(df) == 0) next
  p <- ggplot(df, aes(x = curve_x, y = value, color = method_group)) +
    geom_line(linewidth = 1.0) +
    geom_point(size = 2) +
    facet_wrap(~metric_label, scales = "free_y", ncol = 1) +
    scale_color_manual(values = method_colors) +
    labs(
      title = sprintf("Suffix Sweep on %s", label),
      x = "Suffix Length",
      y = NULL,
      color = "Method"
    ) +
    base_theme
  file_name <- sprintf("fig_suffix_curve_%s.png", label)
  save_plot(p, file_name, width = 9, height = 10)
  register_figure(
    file_name = file_name,
    title = sprintf("%s 后缀长度扫描曲线", label),
    chart_type = "Line",
    data_scope = label,
    meaning = "展示 suffix_len 对 qaw 时延与质量的影响，用于判断尾部强制保留是否有必要。",
    value = "适合支撑“suffix_len 是稳定性控制参数”的论述。",
    recommend_for_paper = FALSE
  )
}

# 只有在同一图中跨数据集比较时，必须满足 count 与参数范围一致
comparable_suffix_groups <- comparable_groups_df %>%
  filter(curve_mode == "suffix_only", dataset_count > 1)

if (nrow(comparable_suffix_groups) > 0) {
  group_id <- comparable_suffix_groups$compare_group[[1]]
  comparable_files <- curve_file_comparability %>%
    filter(compare_group == group_id) %>%
    pull(file_label)

  suffix_compare_df <- curve_clean %>%
    filter(file_label %in% comparable_files, curve_mode == "suffix_only",
           metric %in% c("avg_total_s", "avg_f1"),
           method_group == "query_aware") %>%
    group_by(dataset, curve_x, metric) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
    mutate(metric_label = metric_pretty[metric])

  p <- ggplot(suffix_compare_df, aes(x = curve_x, y = value, color = dataset)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.4) +
    facet_wrap(~metric_label, scales = "free_y", ncol = 1) +
    labs(
      title = "Cross-Dataset Suffix Comparison",
      x = "Suffix Length",
      y = NULL,
      color = "Dataset"
    ) +
    base_theme
  file_name <- "fig_suffix_compare_cross_dataset.png"
  save_plot(p, file_name, width = 8.5, height = 8)
  register_figure(
    file_name = file_name,
    title = "跨数据集 suffix 对比",
    chart_type = "Line",
    data_scope = "仅使用 count 相同、固定 qaw_ratio 相同、suffix 范围相同的可比组",
    meaning = "比较 musique 与 wikimqa 在同一 suffix 扫描设置下的时延和质量走势。",
    value = "满足你的可比性约束，适合放在论文里说明参数对不同任务的影响是否一致。",
    recommend_for_paper = TRUE,
    paper_order = 6L
  )

  suffix_trade <- curve_clean %>%
    filter(file_label %in% comparable_files, curve_mode == "suffix_only",
           metric %in% c("avg_total_s", "avg_f1"),
           method_group %in% c("query_aware", "full_prefill")) %>%
    group_by(dataset, file_label, curve_x, method_group, metric) %>%
    summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = metric, values_from = value)
  suffix_base <- suffix_trade %>%
    filter(method_group == "full_prefill") %>%
    group_by(dataset, curve_x) %>%
    summarise(base_total = mean(avg_total_s), base_f1 = mean(avg_f1), .groups = "drop")
  suffix_qaw <- suffix_trade %>%
    filter(method_group == "query_aware") %>%
    group_by(dataset, curve_x) %>%
    summarise(avg_total_s = mean(avg_total_s), avg_f1 = mean(avg_f1), .groups = "drop") %>%
    left_join(suffix_base, by = c("dataset", "curve_x")) %>%
    mutate(
      speedup_vs_prefill = base_total / avg_total_s,
      f1_retention = avg_f1 / base_f1
    )

  p <- ggplot(suffix_qaw, aes(x = speedup_vs_prefill, y = f1_retention, color = dataset)) +
    geom_path(linewidth = 1) +
    geom_point(aes(size = curve_x), alpha = 0.9) +
    labs(
      title = "Cross-Dataset Suffix Tradeoff",
      x = "Speedup vs Full Prefill",
      y = "F1 Retention",
      color = "Dataset",
      size = "Suffix Len"
    ) +
    base_theme
  file_name <- "fig_suffix_tradeoff_cross_dataset.png"
  save_plot(p, file_name, width = 8.5, height = 6)
  register_figure(
    file_name = file_name,
    title = "跨数据集 suffix 权衡散点图",
    chart_type = "Scatter Path",
    data_scope = "跨数据集可比 suffix 组",
    meaning = "在相同 suffix 扫描设置下，比较两个数据集的速度-质量平衡曲线。",
    value = "适合用来解释“参数对不同任务的收益差异”。",
    recommend_for_paper = FALSE
  )
}

# 样本级分析：只看非曲线主实验
sample_main <- sample_df_clean %>%
  mutate(
    method_group = case_when(
      method %in% c("query_aware", "qaw_default") ~ "query_aware",
      TRUE ~ method
    )
  ) %>%
  filter(method_group %in% c("full_prefill", "full_reuse", "query_aware"))

for (ds in sort(unique(sample_main$dataset))) {
  ds_df <- sample_main %>% filter(dataset == ds)
  if (nrow(ds_df) == 0) next

  p_total <- ggplot(ds_df, aes(x = method_group, y = total_s, fill = method_group)) +
    geom_boxplot(alpha = 0.75, outlier.alpha = 0.5) +
    scale_fill_manual(values = method_colors, guide = "none") +
    labs(
      title = sprintf("Sample Total Latency Distribution on %s", str_to_title(ds)),
      x = "Method",
      y = "Total Latency (s)"
    ) +
    base_theme
  file_name <- sprintf("fig_sample_box_total_%s.png", ds)
  save_plot(p_total, file_name, width = 8, height = 5.5)
  register_figure(
    file_name = file_name,
    title = sprintf("%s 样本级总时延箱线图", ds),
    chart_type = "Boxplot",
    data_scope = sprintf("%s 非曲线样本级结果", ds),
    meaning = "展示不同方法在样本层面的总时延分布，而不是只看平均值。",
    value = "适合说明方法稳定性和离群样本情况。",
    recommend_for_paper = FALSE
  )

  if (all(is.na(ds_df$f1))) next
  p_f1 <- ggplot(ds_df, aes(x = method_group, y = f1, fill = method_group)) +
    geom_boxplot(alpha = 0.75, outlier.alpha = 0.5) +
    scale_fill_manual(values = method_colors, guide = "none") +
    labs(
      title = sprintf("Sample F1 Distribution on %s", str_to_title(ds)),
      x = "Method",
      y = "F1"
    ) +
    base_theme
  file_name <- sprintf("fig_sample_box_f1_%s.png", ds)
  save_plot(p_f1, file_name, width = 8, height = 5.5)
  register_figure(
    file_name = file_name,
    title = sprintf("%s 样本级 F1 箱线图", ds),
    chart_type = "Boxplot",
    data_scope = sprintf("%s 非曲线样本级结果", ds),
    meaning = "比较 full_prefill、full_reuse 和 query_aware 在样本层面的质量分布差异。",
    value = "适合配合均值结果解释“平均值相近，但样本分布可能不同”。",
    recommend_for_paper = FALSE
  )

  paired <- ds_df %>%
    filter(method_group %in% c("full_prefill", "query_aware")) %>%
    group_by(source_file, file_label, sample_idx, method_group) %>%
    summarise(
      total_s = mean(total_s, na.rm = TRUE),
      f1 = mean(f1, na.rm = TRUE),
      recomputed_tokens = mean(recomputed_tokens, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    pivot_wider(names_from = method_group, values_from = c(total_s, f1, recomputed_tokens)) %>%
    mutate(
      latency_gain_s = total_s_full_prefill - total_s_query_aware,
      f1_delta = f1_query_aware - f1_full_prefill
    )

  if (nrow(paired) > 0) {
    p_delta <- ggplot(paired, aes(x = latency_gain_s, y = f1_delta)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
      geom_point(color = method_colors[["query_aware"]], size = 2.6, alpha = 0.8) +
      labs(
        title = sprintf("Sample Delta: QAW vs Full Prefill on %s", str_to_title(ds)),
        x = "Latency Gain (s)",
        y = "F1 Delta"
      ) +
      base_theme
    file_name <- sprintf("fig_sample_delta_scatter_%s.png", ds)
    save_plot(p_delta, file_name, width = 7.5, height = 5.5)
    register_figure(
      file_name = file_name,
      title = sprintf("%s 样本级增益散点图", ds),
      chart_type = "Scatter",
      data_scope = sprintf("%s 非曲线样本级结果", ds),
      meaning = "横轴是 query_aware 相对 full_prefill 的时延收益，纵轴是质量变化，四象限可直接解释收益与代价。",
      value = "非常适合论文分析“哪些样本既加速又不降质，哪些样本会退化”。",
      recommend_for_paper = ds == "musique",
      paper_order = ifelse(ds == "musique", 7L, NA_integer_)
    )

    p_hist <- ggplot(paired, aes(x = latency_gain_s)) +
      geom_histogram(bins = 12, fill = method_colors[["query_aware"]], color = "white") +
      labs(
        title = sprintf("Latency Gain Histogram on %s", str_to_title(ds)),
        x = "Latency Gain (s)",
        y = "Sample Count"
      ) +
      base_theme
    file_name <- sprintf("fig_sample_latency_gain_hist_%s.png", ds)
    save_plot(p_hist, file_name, width = 7.5, height = 5)
    register_figure(
      file_name = file_name,
      title = sprintf("%s 时延收益直方图", ds),
      chart_type = "Histogram",
      data_scope = sprintf("%s 非曲线样本级结果", ds),
      meaning = "观察 query_aware 在样本层面的时延收益分布是否集中。",
      value = "适合补充说明收益是否普遍存在，还是主要由少数样本驱动。",
      recommend_for_paper = FALSE
    )

    if (!all(is.na(paired$recomputed_tokens_query_aware))) {
      p_recomp <- ggplot(paired, aes(x = recomputed_tokens_query_aware, y = total_s_query_aware)) +
        geom_point(color = method_colors[["query_aware"]], size = 2.5, alpha = 0.8) +
        labs(
          title = sprintf("Recomputed Tokens vs Total Latency on %s", str_to_title(ds)),
          x = "Recomputed Tokens",
          y = "QAW Total Latency (s)"
        ) +
        base_theme
      file_name <- sprintf("fig_sample_recomputed_vs_latency_%s.png", ds)
      save_plot(p_recomp, file_name, width = 7.5, height = 5.5)
      register_figure(
        file_name = file_name,
        title = sprintf("%s 重算 token 数与时延关系图", ds),
        chart_type = "Scatter",
        data_scope = sprintf("%s 非曲线样本级结果", ds),
        meaning = "观察样本层面上重算 token 数量与 query_aware 实际时延之间的关系。",
        value = "适合支撑“重算规模会影响时延，但不是唯一因素”的分析。",
        recommend_for_paper = FALSE
      )
    }
  }
}

write.csv(figure_registry, file.path(out_dir, "figure_inventory.csv"), row.names = FALSE)

recommended_figs <- figure_registry %>%
  filter(recommend_for_paper) %>%
  arrange(paper_order, file_name)

inventory_lines <- c(
  "# 图表清单",
  "",
  "本清单由 `graduation_project/analyse/analyze_outputs_revised_0416.R` 自动生成。",
  "",
  "## 分析规则",
  "",
  "- 旧版 curve 输出文件会剔除文件内第一组 `run_summary`；带有独立 hot baseline 覆盖的新流程不再机械丢首组。",
  "- 若存在匹配的 standalone hot baseline 文件，则优先用其覆盖目标文件中的 `full_prefill/full_reuse` 指标；`query_aware` 保持原输出。",
  "- 只有在 `count` 和关键参数范围一致时，才允许把不同数据集放在同一张图中比较。",
  "- `outputs/0416_sum/` 中的 orchestration 日志和失败/无指标文件不会进入正式分析。",
  "",
  "## 全部图表",
  ""
)

for (i in seq_len(nrow(figure_registry))) {
  row <- figure_registry[i, ]
  inventory_lines <- c(
    inventory_lines,
    sprintf("### %s", row$file_name),
    sprintf("- 标题：%s", row$title),
    sprintf("- 类型：%s", row$chart_type),
    sprintf("- 数据范围：%s", row$data_scope),
    sprintf("- 意义：%s", row$meaning),
    sprintf("- 价值：%s", row$value),
    ""
  )
}

inventory_lines <- c(
  inventory_lines,
  "## 适合放入论文的图表建议",
  ""
)

if (nrow(recommended_figs) > 0) {
  for (i in seq_len(nrow(recommended_figs))) {
    row <- recommended_figs[i, ]
    inventory_lines <- c(
      inventory_lines,
      sprintf("%d. `%s`：%s", i, row$file_name, row$title)
    )
  }
} else {
  inventory_lines <- c(inventory_lines, "- 当前未标记推荐图。")
}

inventory_lines <- c(
  inventory_lines,
  "",
  "### 推荐放图逻辑顺序",
  "",
  "1. 先放单数据集主实验方法对比图，交代全文的基线格局。",
  "2. 再放核心 ratio 曲线，解释 query_aware 的参数敏感性。",
  "3. 然后放速度-质量权衡图，突出“不是单看时延，也不是单看 F1”。",
  "4. 接着放 suffix 对比图，说明尾部保留机制的作用。",
  "5. 最后放样本级增益散点图，分析方法收益和退化案例的分布。",
  ""
)

writeLines(inventory_lines, file.path(out_dir, "figure_inventory.md"))

message("Re-analysis complete: ", out_dir)
