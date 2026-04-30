#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
})

`%||%` <- function(a, b) {
  if (is.null(a) || length(a) == 0) b else a
}

script_path_from_args <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- args[startsWith(args, "--file=")]
  if (length(hit) == 0) return(NA_character_)
  normalizePath(sub("^--file=", "", hit[[1]]), mustWork = FALSE)
}

trailing_arg_value <- function(name) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) return(NULL)

  flag <- paste0("--", name)
  inline_prefix <- paste0(flag, "=")
  inline_hit <- args[startsWith(args, inline_prefix)]
  if (length(inline_hit) > 0) {
    return(sub(paste0("^", inline_prefix), "", inline_hit[[length(inline_hit)]]))
  }

  flag_idx <- match(flag, args)
  if (!is.na(flag_idx) && flag_idx < length(args)) {
    return(args[[flag_idx + 1L]])
  }

  NULL
}

requested_qaw_ratio <- function() {
  raw <- trailing_arg_value("qaw-ratio") %||% trailing_arg_value("ratio")
  if (is.null(raw)) return(NA_real_)
  value <- suppressWarnings(as.numeric(raw))
  if (!is.finite(value)) {
    stop("Invalid qaw-ratio argument: ", raw)
  }
  round(value, 4)
}

repo_root_from_script <- function() {
  script_path <- script_path_from_args()
  if (!is.na(script_path) && file.exists(script_path)) {
    return(normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE))
  }
  normalizePath(getwd(), mustWork = FALSE)
}

as_num <- function(x) {
  if (is.null(x) || length(x) == 0) return(NA_real_)
  suppressWarnings(as.numeric(unlist(x, use.names = FALSE)[[1]]))
}

as_int <- function(x) {
  if (is.null(x) || length(x) == 0) return(NA_integer_)
  suppressWarnings(as.integer(unlist(x, use.names = FALSE)[[1]]))
}

as_chr <- function(x) {
  if (is.null(x) || length(x) == 0) return(NA_character_)
  as.character(unlist(x, use.names = FALSE)[[1]])
}

safe_parse_json <- function(line) {
  line <- trimws(line)
  if (!startsWith(line, "{")) return(NULL)
  tryCatch(fromJSON(line, simplifyVector = FALSE), error = function(e) NULL)
}

infer_dataset <- function(file_name, dataset_field = NULL) {
  x <- tolower(paste(file_name, dataset_field %||% "", collapse = " "))
  if (str_detect(x, "musique")) return("musique")
  if (str_detect(x, "wikimqa")) return("wikimqa")
  if (str_detect(x, "cmrc")) return("cmrc")
  if (str_detect(x, "samsum")) return("samsum")
  "unknown"
}

infer_model <- function(model_field, file_name = "") {
  x <- paste(model_field %||% "", file_name %||% "")
  if (str_detect(x, fixed("Qwen2.5-1.5B"))) return("Qwen2.5-1.5B")
  if (str_detect(x, fixed("Yi-6B"))) return("Yi-6B")
  out <- basename(as_chr(model_field))
  if (is.na(out) || out == "") "unknown" else out
}

extract_summary_row <- function(ev, meta, file_path, summary_order) {
  dataset <- infer_dataset(basename(file_path), meta$dataset %||% meta$dataset_name)
  model <- infer_model(meta$model %||% meta$model_name, basename(file_path))
  tibble(
    source_file = basename(file_path),
    source_path = normalizePath(file_path, mustWork = FALSE),
    summary_order = summary_order,
    script = as_chr(meta$script),
    dataset = dataset,
    model = model,
    is_curve_summary = !is.null(ev$recomp_ratio),
    qaw_ratio = as_num(ev$recomp_ratio %||% meta$qaw_ratio),
    suffix_len = as_num(ev$suffix_len),
    sample_count = as_int(ev$sample_count %||% meta$count),
    full_reuse_f1 = as_num(ev$full_reuse_avg_f1),
    query_aware_f1 = as_num(ev$query_aware_avg_f1),
    full_prefill_f1 = as_num(ev$full_prefill_avg_f1),
    full_reuse_total_s = as_num(ev$full_reuse_avg_total_s),
    query_aware_total_s = as_num(ev$query_aware_avg_total_s),
    full_prefill_total_s = as_num(ev$full_prefill_avg_total_s),
    full_reuse_ttft_s = as_num(ev$full_reuse_avg_ttft_s),
    query_aware_ttft_s = as_num(ev$query_aware_avg_ttft_s),
    full_prefill_ttft_s = as_num(ev$full_prefill_avg_ttft_s),
    qaw_true_recompute_count = as_int(ev$qaw_true_recompute_count),
    started_at = as_chr(meta$started_at),
    ended_at = as_chr(ev$ended_at)
  )
}

extract_sample_method <- function(ev, meta, file_path, method_key, method_label) {
  node <- ev[[method_key]]
  if (is.null(node) || !is.list(node)) return(NULL)
  dataset <- infer_dataset(basename(file_path), meta$dataset %||% meta$dataset_name)
  model <- infer_model(meta$model %||% meta$model_name, basename(file_path))
  tibble(
    source_file = basename(file_path),
    source_path = normalizePath(file_path, mustWork = FALSE),
    script = as_chr(meta$script),
    dataset = dataset,
    model = model,
    qaw_ratio = as_num(meta$qaw_ratio),
    sample_idx = as_int(ev$sample_idx),
    chunk_num = as_int(ev$chunk_num),
    method_key = method_key,
    method = method_label,
    f1 = as_num(node$f1),
    total_s = as_num(node$total_s),
    ttft_s = as_num(node$ttft_s),
    recomputed_tokens = as_num(node$recomputed_tokens),
    true_recompute = as.logical(node$true_recompute %||% NA)
  )
}

parse_output_file <- function(file_path) {
  lines <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  meta <- list()
  summary_rows <- list()
  sample_rows <- list()
  summary_order <- 0L

  for (line in lines) {
    ev <- safe_parse_json(line)
    if (is.null(ev) || is.null(ev$event)) next

    if (identical(ev$event, "run_start")) {
      meta <- ev
      next
    }

    if (identical(ev$event, "run_summary")) {
      summary_order <- summary_order + 1L
      summary_rows[[length(summary_rows) + 1L]] <-
        extract_summary_row(ev, meta, file_path, summary_order)
      next
    }

    if (identical(ev$event, "sample_result")) {
      sample_rows <- c(
        sample_rows,
        list(extract_sample_method(ev, meta, file_path, "full_reuse", "Full Reuse")),
        list(extract_sample_method(ev, meta, file_path, "query_aware", "Query-Aware")),
        list(extract_sample_method(ev, meta, file_path, "full_prefill", "Full Prefill"))
      )
    }
  }

  list(
    summary = bind_rows(compact(summary_rows)),
    sample = bind_rows(compact(sample_rows))
  )
}

method_colors <- c(
  "Full Reuse" = "#0072B2",
  "Query-Aware" = "#009E73",
  "Full Prefill" = "#D55E00"
)
method_shapes <- c(
  "Full Reuse" = 17,
  "Query-Aware" = 16,
  "Full Prefill" = 15
)
method_linetypes <- c(
  "Full Reuse" = "dashed",
  "Query-Aware" = "solid",
  "Full Prefill" = "dotdash"
)
dataset_levels <- c("musique", "wikimqa", "samsum")
dataset_labels <- c(
  musique = "Musique",
  wikimqa = "2WikiMQA",
  samsum = "SAMSum"
)
dataset_score_labels <- c(
  musique = "Musique (F1)",
  wikimqa = "2WikiMQA (F1)",
  samsum = "SAMSum (Rouge-L)"
)
model_levels <- c("Yi-6B", "Qwen2.5-1.5B")

decorate_common <- function(p) {
  p +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5),
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "#F2F2F2", color = "#D9D9D9"),
      legend.position = "bottom"
    )
}

f1_axis_limits <- function(values, window = 0.1) {
  finite_values <- values[is.finite(values)]
  if (length(finite_values) == 0) return(c(0, 1))

  center <- median(finite_values, na.rm = TRUE)
  lower <- floor(center / window) * window
  upper <- lower + window

  if (upper > 1) {
    upper <- 1
    lower <- max(0, upper - window)
  }

  c(round(max(0, lower), 4), round(min(1, upper), 4))
}

f1_axis_breaks <- function(limits) {
  round(seq(limits[[1]], limits[[2]], length.out = 6), 2)
}

axis_breaks_from_limits <- function(limits, n = 4) {
  pretty(limits, n = n)
}

extract_legend_grob <- function(plot_obj) {
  components <- suppressWarnings(
    cowplot::get_plot_component(plot_obj, "guide-box", return_all = TRUE)
  )
  legend_idx <- which(vapply(components, function(x) inherits(x, "gtable"), logical(1)))
  if (length(legend_idx) == 0) {
    stop("Failed to extract legend grob from plot.")
  }
  components[[legend_idx[[1]]]]
}

axis_limits_with_padding <- function(values, pad_ratio = 0.06, min_span = NULL) {
  finite_values <- values[is.finite(values)]
  if (length(finite_values) == 0) return(c(0, 1))

  lower <- min(finite_values, na.rm = TRUE)
  upper <- max(finite_values, na.rm = TRUE)
  span <- upper - lower

  if (!is.finite(span) || span <= 0) {
    span <- if (is.null(min_span)) max(abs(lower) * 0.1, 0.1) else min_span
  }
  if (!is.null(min_span)) {
    span <- max(span, min_span)
  }

  pad <- span * pad_ratio
  c(round(lower - pad, 4), round(upper + pad, 4))
}

metric_labels <- c(
  f1 = "F1",
  total_s = "Total Latency (s)",
  ttft_s = "TTFT (s)"
)

score_metric_label <- function(dataset_name) {
  if (identical(dataset_name, "samsum")) "Rouge-L" else "F1"
}

average_score_axis_label <- function(dataset_name) {
  paste("Average", score_metric_label(dataset_name))
}

repo_root <- repo_root_from_script()
outputs_dir <- file.path(repo_root, "outputs")
out_dir <- file.path(repo_root, "graduation_project", "analyse", "qaw_visualizations")
fig_dir <- file.path(out_dir, "figures")
script_dir <- dirname(script_path_from_args())
if (is.na(script_dir) || !dir.exists(script_dir)) {
  script_dir <- file.path(repo_root, "scripts")
}
doc_path <- file.path(script_dir, "qaw_visualization_report.md")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

output_files <- list.files(outputs_dir, pattern = "\\.output$", full.names = TRUE, recursive = FALSE)
if (length(output_files) == 0) {
  stop("No .output files found under: ", outputs_dir)
}

parsed <- map(sort(output_files), parse_output_file)
summary_raw <- bind_rows(map(parsed, "summary"))
sample_raw <- bind_rows(map(parsed, "sample"))

curve_df <- summary_raw %>%
  filter(
    dataset %in% dataset_levels,
    model %in% model_levels,
    is_curve_summary,
    !is.na(qaw_ratio),
    !is.na(query_aware_f1),
    !is.na(query_aware_total_s),
    !is.na(query_aware_ttft_s)
  ) %>%
  mutate(
    dataset_label = factor(unname(dataset_labels[dataset]), levels = unname(dataset_labels)),
    dataset_score_label = factor(unname(dataset_score_labels[dataset]), levels = unname(dataset_score_labels)),
    model_label = factor(model, levels = model_levels),
    qaw_ratio = round(qaw_ratio, 4)
  ) %>%
  filter(
    dataset != "musique" |
      model != "Yi-6B" |
      source_file == "202604212001_musique_curve.output"
  )

if (nrow(curve_df) == 0) {
  stop("No usable curve run_summary rows were parsed from outputs/*.output.")
}

curve_aug <- curve_df %>%
  mutate(
    f1_retention_vs_prefill = query_aware_f1 / pmax(full_prefill_f1, 1e-9),
    f1_delta_vs_prefill = query_aware_f1 - full_prefill_f1,
    f1_delta_vs_reuse = query_aware_f1 - full_reuse_f1,
    speedup_vs_prefill = full_prefill_total_s / query_aware_total_s,
    latency_ratio_vs_prefill = query_aware_total_s / full_prefill_total_s,
    latency_overhead_pct = (latency_ratio_vs_prefill - 1) * 100,
    ttft_ratio_vs_prefill = query_aware_ttft_s / full_prefill_ttft_s,
    balanced_score = 0.65 * pmin(f1_retention_vs_prefill, 1.2) +
      0.35 * pmin(speedup_vs_prefill, 1.5)
  )

ratio_scores <- curve_aug %>%
  group_by(qaw_ratio) %>%
  summarise(
    group_count = n_distinct(paste(model, dataset, sep = "::")),
    avg_f1_retention = mean(f1_retention_vs_prefill, na.rm = TRUE),
    avg_speedup_vs_prefill = mean(speedup_vs_prefill, na.rm = TRUE),
    avg_latency_overhead_pct = mean(latency_overhead_pct, na.rm = TRUE),
    avg_query_aware_f1 = mean(query_aware_f1, na.rm = TRUE),
    avg_query_aware_total_s = mean(query_aware_total_s, na.rm = TRUE),
    avg_balanced_score = mean(balanced_score, na.rm = TRUE),
    .groups = "drop"
  )

target_groups <- length(model_levels) * length(dataset_levels)
speedup_floor <- 0.87
forced_ratio <- requested_qaw_ratio()
eligible_ratios <- ratio_scores %>%
  filter(
    group_count == target_groups,
    qaw_ratio > 0,
    qaw_ratio < 1,
    avg_speedup_vs_prefill >= speedup_floor
  ) %>%
  arrange(desc(avg_f1_retention), desc(avg_balanced_score), qaw_ratio)

available_ratios <- sort(unique(ratio_scores$qaw_ratio))
selected_ratio <- if (is.finite(forced_ratio)) {
  exact_hit <- available_ratios[abs(available_ratios - forced_ratio) < 1e-9]
  if (length(exact_hit) == 0) {
    stop(
      sprintf(
        "Requested qaw-ratio %.4f is not available. Available ratios: %s",
        forced_ratio,
        paste(sprintf("%.4f", available_ratios), collapse = ", ")
      )
    )
  }
  exact_hit[[1]]
} else if (nrow(eligible_ratios) > 0) {
  eligible_ratios$qaw_ratio[[1]]
} else {
  ratio_scores %>%
    filter(group_count == max(group_count)) %>%
    arrange(desc(avg_balanced_score), qaw_ratio) %>%
    pull(qaw_ratio) %>%
    .[[1]]
}

model_selected_ratios_manual <- tibble::tribble(
  ~model,         ~selected_ratio,
  "Yi-6B",        0.60,
  "Qwen2.5-1.5B", 0.45
)

model_selected_ratios_auto <- tibble(
  model = model_levels,
  selected_ratio = rep(selected_ratio, length(model_levels))
)

model_selected_ratios <- model_selected_ratios_auto %>%
  left_join(model_selected_ratios_manual, by = "model", suffix = c("_auto", "_manual")) %>%
  transmute(
    model,
    selected_ratio = coalesce(selected_ratio_manual, selected_ratio_auto),
    model_label = factor(model, levels = model_levels)
  )

missing_model_ratios <- model_selected_ratios %>%
  filter(!selected_ratio %in% available_ratios)
if (nrow(missing_model_ratios) > 0) {
  stop(
    sprintf(
      "Model-specific selected ratios are unavailable: %s. Available ratios: %s",
      paste(sprintf("%s=%.4f", missing_model_ratios$model, missing_model_ratios$selected_ratio), collapse = ", "),
      paste(sprintf("%.4f", available_ratios), collapse = ", ")
    )
  )
}

method_points <- bind_rows(
  curve_df %>%
    transmute(
      source_file, dataset, dataset_label, dataset_score_label, model, model_label, qaw_ratio,
      method_key = "full_reuse", method = "Full Reuse",
      f1 = full_reuse_f1, total_s = full_reuse_total_s, ttft_s = full_reuse_ttft_s
    ),
  curve_df %>%
    transmute(
      source_file, dataset, dataset_label, dataset_score_label, model, model_label, qaw_ratio,
      method_key = "query_aware", method = "Query-Aware",
      f1 = query_aware_f1, total_s = query_aware_total_s, ttft_s = query_aware_ttft_s
    ),
  curve_df %>%
    transmute(
      source_file, dataset, dataset_label, dataset_score_label, model, model_label, qaw_ratio,
      method_key = "full_prefill", method = "Full Prefill",
      f1 = full_prefill_f1, total_s = full_prefill_total_s, ttft_s = full_prefill_ttft_s
    )
) %>%
  mutate(method = factor(method, levels = names(method_colors)))

method_metric_long <- method_points %>%
  pivot_longer(c(f1, total_s, ttft_s), names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = factor(unname(metric_labels[metric]), levels = unname(metric_labels)),
    metric_display_label = case_when(
      metric == "f1" & dataset == "samsum" ~ "Rouge-L",
      metric == "f1" ~ "F1",
      TRUE ~ unname(metric_labels[metric])
    )
  )

query_points <- curve_aug %>%
  transmute(
    source_file, dataset, dataset_label, dataset_score_label, model, model_label, qaw_ratio,
    f1 = query_aware_f1,
    total_s = query_aware_total_s,
    ttft_s = query_aware_ttft_s,
    f1_retention_vs_prefill,
    speedup_vs_prefill,
    latency_overhead_pct,
    balanced_score
  )

baseline_points <- method_points %>%
  filter(method != "Query-Aware") %>%
  group_by(model, dataset, dataset_score_label, method) %>%
  slice(1) %>%
  ungroup()

figure_registry <- tibble(
  file_name = character(),
  title = character(),
  chart_type = character(),
  data_scope = character(),
  meaning = character(),
  paper_use = character()
)

register_figure <- function(file_name, title, chart_type, data_scope, meaning, paper_use) {
  figure_registry <<- bind_rows(
    figure_registry,
    tibble(
      file_name = file_name,
      title = title,
      chart_type = chart_type,
      data_scope = data_scope,
      meaning = meaning,
      paper_use = paper_use
    )
  )
}

save_plot <- function(plot_obj, file_name, width, height, title, chart_type,
                      data_scope, meaning, paper_use) {
  ggsave(
    filename = file.path(fig_dir, file_name),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )
  register_figure(file_name, title, chart_type, data_scope, meaning, paper_use)
}

selected_points <- method_points %>%
  inner_join(model_selected_ratios %>% select(model, selected_ratio), by = "model") %>%
  filter(abs(qaw_ratio - selected_ratio) < 1e-9) %>%
  select(-selected_ratio)

# # Per-panel axis configuration for Figure 1.
# fig1_panel_limits_auto <- selected_points %>%
#   mutate(model = as.character(model), dataset = as.character(dataset)) %>%
#   group_by(model, dataset) %>%
#   summarise(
#     x_min = axis_limits_with_padding(total_s, min_span = 0.5)[[1]],
#     x_max = axis_limits_with_padding(total_s, min_span = 0.5)[[2]],
#     y_min = f1_axis_limits(f1)[[1]],
#     y_max = f1_axis_limits(f1)[[2]],
#     .groups = "drop"
#   ) %>%
#   mutate(
#     model = factor(model, levels = model_levels),
#     dataset = factor(dataset, levels = dataset_levels)
#   ) %>%
#   arrange(model, dataset) %>%
#   mutate(
#     model = as.character(model),
#     dataset = as.character(dataset)
#   )
#
# # Manual overrides for Figure 1 panel axes. Fill any x_min/x_max/y_min/y_max you want to fix.
# fig1_panel_limits_manual <- tibble::tribble(
#   ~model,          ~dataset,   ~x_min,   ~x_max,   ~y_min,   ~y_max,
#   "Yi-6B",         "musique",  NA_real_, NA_real_, NA_real_, NA_real_,
#   "Yi-6B",         "wikimqa",  NA_real_, NA_real_, NA_real_, NA_real_,
#   "Yi-6B",         "samsum",   NA_real_, NA_real_, NA_real_, NA_real_,
#   "Qwen2.5-1.5B",  "musique",  NA_real_, NA_real_, NA_real_, NA_real_,
#   "Qwen2.5-1.5B",  "wikimqa",  NA_real_, NA_real_, NA_real_, NA_real_,
#   "Qwen2.5-1.5B",  "samsum",   NA_real_, NA_real_, NA_real_, NA_real_
# )
#
# fig1_panel_limits <- fig1_panel_limits_auto %>%
#   left_join(fig1_panel_limits_manual, by = c("model", "dataset"), suffix = c("_auto", "_manual")) %>%
#   transmute(
#     model,
#     dataset,
#     x_min = coalesce(x_min_manual, x_min_auto),
#     x_max = coalesce(x_max_manual, x_max_auto),
#     y_min = coalesce(y_min_manual, y_min_auto),
#     y_max = coalesce(y_max_manual, y_max_auto)
#   )
#
# fig1_legend_plot <- ggplot(selected_points, aes(x = total_s, y = f1, color = method, shape = method)) +
#   geom_point(size = 3.6, stroke = 1.0) +
#   scale_color_manual(values = method_colors) +
#   scale_shape_manual(values = method_shapes) +
#   labs(color = "Method", shape = "Method") +
#   theme_bw(base_size = 12) +
#   theme(legend.position = "bottom")
#
# fig1_legend <- extract_legend_grob(fig1_legend_plot)
#
# build_fig1_panel <- function(model_name, dataset_name, x_min, x_max, y_min, y_max) {
#   panel_df <- selected_points %>%
#     filter(as.character(.data$model) == model_name, as.character(.data$dataset) == dataset_name)
#
#   x_limits <- c(as.numeric(x_min), as.numeric(x_max))
#   y_limits <- c(as.numeric(y_min), as.numeric(y_max))
#
#   if (!all(is.finite(c(x_limits, y_limits)))) {
#     stop(
#       sprintf(
#         "Invalid Figure 1 axis config for %s / %s: x_min=%s, x_max=%s, y_min=%s, y_max=%s",
#         model_name, dataset_name, x_min, x_max, y_min, y_max
#       )
#     )
#   }
#
#   p <- ggplot(panel_df, aes(x = total_s, y = f1, color = method, shape = method)) +
#     geom_point(size = 3.6, stroke = 1.0) +
#     coord_cartesian(xlim = x_limits, ylim = y_limits) +
#     scale_color_manual(values = method_colors) +
#     scale_shape_manual(values = method_shapes) +
#     scale_x_continuous(breaks = axis_breaks_from_limits(x_limits)) +
#     scale_y_continuous(breaks = f1_axis_breaks(y_limits)) +
#     labs(
#       title = sprintf("%s | %s", model_name, dataset_score_labels[[dataset_name]]),
#       x = "Average Total Latency (s)",
#       y = average_score_axis_label(dataset_name)
#     )
#
#   decorate_common(p) +
#     theme(
#       legend.position = "none",
#       plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
#       axis.title = element_text(size = 10)
#     )
# }
#
# fig1_panels <- purrr::pmap(
#   fig1_panel_limits %>% select(model, dataset, x_min, x_max, y_min, y_max),
#   build_fig1_panel
# )
#
# fig1_grid <- cowplot::plot_grid(plotlist = fig1_panels, ncol = length(dataset_levels), align = "hv")
# p1 <- cowplot::plot_grid(
#   cowplot::ggdraw() +
#     cowplot::draw_label(
#       "Latency-Quality Comparison",
#       fontface = "bold",
#       x = 0.5,
#       hjust = 0.5,
#       size = 14
#     ) +
#     cowplot::draw_label(
#       "Each panel compares Full Reuse, Query-Aware, and Full Prefill",
#       x = 0.5,
#       y = 0.2,
#       hjust = 0.5,
#       size = 11
#     ),
#   fig1_grid,
#   fig1_legend,
#   ncol = 1,
#   rel_heights = c(0.12, 1, 0.08)
# )
# save_plot(
#   p1,
#   "fig01_best_ratio_latency_f1_grid.png",
#   12,
#   7,
#   "Latency-Quality Comparison",
#   "2 x 3 scatter grid",
#   "Current root outputs/*.output, model-specific selected settings",
#   "Compares total generation latency and F1 or Rouge-L for the three methods across two models and three datasets.",
#   "Main figure candidate for Chapter 4 method comparison."
# )

# Per-panel axis configuration for Figure 17.
fig17_panel_limits_auto <- selected_points %>%
  mutate(model = as.character(model), dataset = as.character(dataset)) %>%
  group_by(model, dataset) %>%
  summarise(
    x_min = axis_limits_with_padding(ttft_s, min_span = 0.05)[[1]],
    x_max = axis_limits_with_padding(ttft_s, min_span = 0.05)[[2]],
    y_min = f1_axis_limits(f1)[[1]],
    y_max = f1_axis_limits(f1)[[2]],
    .groups = "drop"
  ) %>%
  mutate(
    model = factor(model, levels = model_levels),
    dataset = factor(dataset, levels = dataset_levels)
  ) %>%
  arrange(model, dataset) %>%
  mutate(
    model = as.character(model),
    dataset = as.character(dataset)
  )

# Manual overrides for Figure 17 panel axes. Fill any x_min/x_max/y_min/y_max you want to fix.
fig17_panel_limits_manual <- tibble::tribble(
  ~model,          ~dataset,   ~x_min,   ~x_max,   ~y_min,   ~y_max,
  "Yi-6B",         "musique",  NA_real_, NA_real_, 0.20, 0.26,
  "Yi-6B",         "wikimqa",  NA_real_, NA_real_, 0.20, 0.30,
  "Yi-6B",         "samsum",   NA_real_, NA_real_, 0, 0.30,
  "Qwen2.5-1.5B",  "musique",  NA_real_, NA_real_, 0, 0.26,
  "Qwen2.5-1.5B",  "wikimqa",  NA_real_, NA_real_, 0, 0.50,
  "Qwen2.5-1.5B",  "samsum",   NA_real_, NA_real_, 0, 0.40
)

fig17_panel_limits <- fig17_panel_limits_auto %>%
  left_join(fig17_panel_limits_manual, by = c("model", "dataset"), suffix = c("_auto", "_manual")) %>%
  transmute(
    model,
    dataset,
    x_min = coalesce(x_min_manual, x_min_auto),
    x_max = coalesce(x_max_manual, x_max_auto),
    y_min = coalesce(y_min_manual, y_min_auto),
    y_max = coalesce(y_max_manual, y_max_auto)
  )

fig17_legend_plot <- ggplot(selected_points, aes(x = ttft_s, y = f1, color = method, shape = method)) +
  geom_point(size = 3.6, stroke = 1.0) +
  scale_color_manual(values = method_colors) +
  scale_shape_manual(values = method_shapes) +
  labs(color = "Method", shape = "Method") +
  theme_bw(base_size = 12) +
  theme(legend.position = "bottom")

fig17_legend <- extract_legend_grob(fig17_legend_plot)

build_fig17_panel <- function(model_name, dataset_name, x_min, x_max, y_min, y_max) {
  panel_df <- selected_points %>%
    filter(as.character(.data$model) == model_name, as.character(.data$dataset) == dataset_name)

  x_limits <- c(as.numeric(x_min), as.numeric(x_max))
  y_limits <- c(as.numeric(y_min), as.numeric(y_max))

  if (!all(is.finite(c(x_limits, y_limits)))) {
    stop(
      sprintf(
        "Invalid Figure 17 axis config for %s / %s: x_min=%s, x_max=%s, y_min=%s, y_max=%s",
        model_name, dataset_name, x_min, x_max, y_min, y_max
      )
    )
  }

  p <- ggplot(panel_df, aes(x = ttft_s, y = f1, color = method, shape = method)) +
    geom_point(size = 3.6, stroke = 1.0) +
    coord_cartesian(xlim = x_limits, ylim = y_limits) +
    scale_color_manual(values = method_colors) +
    scale_shape_manual(values = method_shapes) +
    scale_x_continuous(breaks = axis_breaks_from_limits(x_limits)) +
    scale_y_continuous(breaks = f1_axis_breaks(y_limits)) +
    labs(
      title = sprintf("%s | %s", model_name, dataset_score_labels[[dataset_name]]),
      x = "Average TTFT (s)",
      y = average_score_axis_label(dataset_name)
    )

  decorate_common(p) +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 10)
    )
}

fig17_panels <- purrr::pmap(
  fig17_panel_limits %>% select(model, dataset, x_min, x_max, y_min, y_max),
  build_fig17_panel
)

fig17_grid <- cowplot::plot_grid(plotlist = fig17_panels, ncol = length(dataset_levels), align = "hv")
p17 <- cowplot::plot_grid(
  cowplot::ggdraw() +
    cowplot::draw_label(
      "TTFT-Quality Comparison",
      fontface = "bold",
      x = 0.5,
      hjust = 0.5,
      size = 14
    ) +
    cowplot::draw_label(
      "Each panel compares Full Reuse, Query-Aware, and Full Prefill",
      x = 0.5,
      y = 0.2,
      hjust = 0.5,
      size = 11
    ),
  fig17_grid,
  fig17_legend,
  ncol = 1,
  rel_heights = c(0.12, 1, 0.08)
)
save_plot(
  p17,
  "fig17_best_ratio_ttft_f1_grid.png",
  12,
  7,
  "TTFT-Quality Comparison",
  "2 x 3 scatter grid",
  "Current root outputs/*.output, model-specific selected settings",
  "Compares TTFT and F1 or Rouge-L for the three methods across two models and three datasets.",
  "Useful as a companion figure to the total-latency comparison when Chapter 4 discusses first-token delay."
)

bar_method_order <- c("Full Reuse", "Query-Aware", "Full Prefill")
bar_model_order <- c("Qwen2.5-1.5B", "Yi-6B")
bar_x_levels <- c(
  "Qwen\nReuse", "Qwen\nQAW", "Qwen\nPrefill",
  "Yi\nReuse", "Yi\nQAW", "Yi\nPrefill"
)

bar_chart_df <- selected_points %>%
  mutate(
    model = factor(model, levels = bar_model_order),
    method = factor(method, levels = bar_method_order),
    x_label = case_when(
      model == "Qwen2.5-1.5B" & method == "Full Reuse" ~ "Qwen\nReuse",
      model == "Qwen2.5-1.5B" & method == "Query-Aware" ~ "Qwen\nQAW",
      model == "Qwen2.5-1.5B" & method == "Full Prefill" ~ "Qwen\nPrefill",
      model == "Yi-6B" & method == "Full Reuse" ~ "Yi\nReuse",
      model == "Yi-6B" & method == "Query-Aware" ~ "Yi\nQAW",
      TRUE ~ "Yi\nPrefill"
    ),
    x_label = factor(x_label, levels = bar_x_levels)
  ) %>%
  arrange(dataset, model, method)

bar_score_upper <- min(1, max(bar_chart_df$f1, na.rm = TRUE) * 1.18)

p18 <- ggplot(bar_chart_df, aes(x = x_label, y = f1, fill = method)) +
  geom_col(width = 0.72, color = "#333333", linewidth = 0.35) +
  geom_vline(xintercept = 3.5, linewidth = 0.35, linetype = "dashed", color = "#9A9A9A") +
  geom_text(
    aes(label = sprintf("%.2f", f1)),
    angle = 0,
    vjust = -0.35,
    hjust = 0.5,
    size = 3.7
  ) +
  facet_wrap(~ dataset_score_label, nrow = 1) +
  coord_cartesian(ylim = c(0, bar_score_upper), clip = "off") +
  scale_fill_manual(
    values = c(
      "Full Reuse" = "#F4F4F4",
      "Query-Aware" = "#F2C79A",
      "Full Prefill" = "#2C7FB8"
    )
  ) +
  labs(
    title = "Selected Setting Score Bars",
    subtitle = "Each panel compares Qwen and Yi under Full Reuse, Query-Aware, and Full Prefill",
    x = NULL,
    y = "Score (F1 / Rouge-L)",
    fill = "Method"
  )
p18 <- decorate_common(p18) +
  theme(
    plot.margin = margin(10, 12, 6, 6),
    axis.text.x = element_text(size = 10, face = "bold"),
    axis.title.y = element_text(face = "bold"),
    panel.grid.major.x = element_blank()
  )
save_plot(
  p18,
  "fig18_selected_setting_score_bars.png",
  14,
  5.8,
  "Selected Setting Score Bars",
  "Faceted bar chart",
  "Three datasets, Qwen and Yi under model-specific selected settings",
  "Compares the selected-score levels of full reuse, query-aware, and full prefill across both models within each dataset.",
  "Useful for a compact side-by-side comparison of method quality under the chosen settings."
)

plot_quality_baseline_curves_yi <- function() {
  selected_ratio_for_model <- model_selected_ratios %>%
    filter(model == "Yi-6B") %>%
    pull(selected_ratio) %>%
    .[[1]]

  yi_query_df <- query_points %>%
    filter(model == "Yi-6B") %>%
    transmute(
      dataset,
      dataset_label,
      qaw_ratio,
      method = "Query-Aware",
      value = f1
    )

  yi_baseline_df <- baseline_points %>%
    filter(model == "Yi-6B", method %in% c("Full Reuse", "Full Prefill")) %>%
    transmute(
      dataset,
      dataset_label,
      method,
      baseline_value = f1
    ) %>%
    mutate(method = factor(method, levels = names(method_colors)))

  p <- ggplot(yi_query_df, aes(x = qaw_ratio, y = value, color = method, linetype = method)) +
    geom_hline(
      data = yi_baseline_df,
      aes(yintercept = baseline_value, color = method, linetype = method),
      linewidth = 0.8,
      alpha = 0.95
    ) +
    geom_line(linewidth = 0.85, alpha = 0.95) +
    geom_point(size = 1.8, alpha = 0.95) +
    geom_vline(
      xintercept = selected_ratio_for_model,
      linewidth = 0.35,
      linetype = "dotted",
      color = "#4D4D4D"
    ) +
    facet_wrap(~ dataset_label, scales = "free_y", nrow = 1) +
    scale_color_manual(values = method_colors, breaks = c("Full Reuse", "Query-Aware", "Full Prefill")) +
    scale_linetype_manual(values = method_linetypes, breaks = c("Full Reuse", "Query-Aware", "Full Prefill")) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
    labs(
      title = "Yi-6B Quality Baseline Curves",
      subtitle = "Query-Aware score with Full Reuse and Full Prefill references",
      x = "Re-compute Ratio",
      y = "Score",
      color = "Method",
      linetype = "Method"
    )

  decorate_common(p) +
    theme(
      strip.background = element_rect(fill = "#F2F2F2", color = "#D9D9D9"),
      strip.text.x = element_text(face = "bold")
    )
}

save_plot(
  plot_quality_baseline_curves_yi(),
  "fig02_yi6b_ratio_metric_curves.png",
  13,
  4.6,
  "Yi-6B Quality Baseline Curves",
  "Faceted line chart",
  "Yi-6B, three datasets, score only",
  "Shows how score changes with recompute ratio, with both Full Reuse and Full Prefill shown as references.",
  "Useful for discussing both quality recovery over reuse and the remaining gap to full prefill."
)

plot_speed_baseline_curves_yi <- function() {
  selected_ratio_for_model <- model_selected_ratios %>%
    filter(model == "Yi-6B") %>%
    pull(selected_ratio) %>%
    .[[1]]

  yi_query_df <- query_points %>%
    filter(model == "Yi-6B") %>%
    transmute(
      dataset,
      dataset_label,
      qaw_ratio,
      method = "Query-Aware",
      value = ttft_s
    )

  yi_baseline_df <- baseline_points %>%
    filter(model == "Yi-6B", method %in% c("Full Reuse", "Full Prefill")) %>%
    transmute(
      dataset,
      dataset_label,
      method,
      baseline_value = ttft_s
    ) %>%
    mutate(method = factor(method, levels = names(method_colors)))

  p <- ggplot(yi_query_df, aes(x = qaw_ratio, y = value, color = method, linetype = method)) +
    geom_hline(
      data = yi_baseline_df,
      aes(yintercept = baseline_value, color = method, linetype = method),
      linewidth = 0.8,
      alpha = 0.95
    ) +
    geom_line(linewidth = 0.85, alpha = 0.95) +
    geom_point(size = 1.8, alpha = 0.95) +
    geom_vline(
      xintercept = selected_ratio_for_model,
      linewidth = 0.35,
      linetype = "dotted",
      color = "#4D4D4D"
    ) +
    facet_wrap(~ dataset_label, scales = "free_y", nrow = 1) +
    scale_color_manual(values = method_colors, breaks = c("Full Reuse", "Query-Aware", "Full Prefill")) +
    scale_linetype_manual(values = method_linetypes, breaks = c("Full Reuse", "Query-Aware", "Full Prefill")) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
    labs(
      title = "Yi-6B Speed Baseline Curves",
      subtitle = "Query-Aware TTFT with Full Reuse and Full Prefill references",
      x = "Re-compute Ratio",
      y = "TTFT (s)",
      color = "Method",
      linetype = "Method"
    )

  decorate_common(p) +
    theme(
      strip.background = element_rect(fill = "#F2F2F2", color = "#D9D9D9"),
      strip.text.x = element_text(face = "bold")
    )
}

save_plot(
  plot_speed_baseline_curves_yi(),
  "fig03_yi6b_speed_baseline_curves.png",
  13,
  4.6,
  "Yi-6B Speed Baseline Curves",
  "Faceted line chart",
  "Yi-6B, three datasets, TTFT only",
  "Shows how TTFT changes with recompute ratio, with both Full Reuse and Full Prefill shown as references.",
  "Useful for discussing where Query-Aware sits between the fastest and the slowest first-token baselines."
)
#
# save_plot(
#   plot_metric_curves("Qwen2.5-1.5B", "fig03_qwen25_15b_ratio_metric_curves.png"),
#   "fig03_qwen25_15b_ratio_metric_curves.png",
#   13,
#   9,
#   "Qwen2.5-1.5B QAW Ratio Curves",
#   "Faceted line chart",
#   "Qwen2.5-1.5B, three datasets, qaw-ratio 0.00 to 1.00",
#   "Shows how F1 or Rouge-L, total latency, and TTFT change with qaw-ratio while full reuse and full prefill stay as references.",
#   "Good for explaining the ratio sensitivity of Qwen2.5-1.5B."
# )

## plot_tradeoff_path <- function(model_name) {
#   qdf <- query_points %>% filter(model == model_name) %>% arrange(dataset, qaw_ratio)
#   bdf <- baseline_points %>% filter(model == model_name)
#   selected_ratio_for_model <- model_selected_ratios %>%
#     filter(model == model_name) %>%
#     pull(selected_ratio) %>%
#     .[[1]]
#   hdf <- qdf %>% filter(abs(qaw_ratio - selected_ratio_for_model) < 1e-9)
#   tradeoff_f1_limits <- f1_axis_limits(c(qdf$f1, bdf$f1))
#   p <- ggplot(qdf, aes(x = total_s, y = f1)) +
#     geom_path(color = "#777777", linewidth = 0.55, alpha = 0.8) +
#     geom_point(aes(color = qaw_ratio), size = 2.8, alpha = 0.92) +
#     geom_point(
#       data = hdf,
#       aes(x = total_s, y = f1),
#       inherit.aes = FALSE,
#       shape = 21,
#       fill = "#FFD166",
#       color = "#222222",
#       size = 4.0,
#       stroke = 0.8
#     ) +
#     geom_point(
#       data = bdf,
#       aes(x = total_s, y = f1, shape = method),
#       inherit.aes = FALSE,
#       color = "#222222",
#       fill = "white",
#       size = 3.4,
#       stroke = 1.0
#     ) +
#     facet_wrap(~ dataset_score_label, scales = "free_x", nrow = 2) +
#     coord_cartesian(ylim = tradeoff_f1_limits) +
#     scale_color_gradient(low = "#3B6EA8", high = "#D95F02") +
#     scale_shape_manual(values = method_shapes[c("Full Reuse", "Full Prefill")]) +
#     scale_y_continuous(breaks = f1_axis_breaks(tradeoff_f1_limits)) +
#     labs(
#       title = paste(model_name, "Latency-Quality Trade-off Path"),
#       subtitle = "Yellow marker highlights the selected setting",
#       x = "Average Total Latency (s)",
#       y = "Average Score",
#       color = "QAW Ratio",
#       shape = "Baseline"
#     )
#   decorate_common(p)
# }
#
# save_plot(
#   plot_tradeoff_path("Yi-6B"),
#   "fig04_yi6b_latency_quality_tradeoff_path.png",
#   12,
#   7,
#   "Yi-6B Latency-Quality Trade-off Path",
#   "Trade-off scatter path",
#   "Yi-6B query-aware curve with full reuse/full prefill reference points",
#   "Visualizes where each qaw-ratio sits in the latency-quality plane and whether it approaches the full prefill quality region.",
#   "Useful for discussing the cost of recovering quality."
# )
#
# save_plot(
#   plot_tradeoff_path("Qwen2.5-1.5B"),
#   "fig05_qwen25_15b_latency_quality_tradeoff_path.png",
#   12,
#   7,
#   "Qwen2.5-1.5B Latency-Quality Trade-off Path",
#   "Trade-off scatter path",
#   "Qwen2.5-1.5B query-aware curve with full reuse/full prefill reference points",
#   "Visualizes where each qaw-ratio sits in the latency-quality plane and whether it approaches the full prefill quality region.",
#   "Useful for discussing the cost of recovering quality."
# )

plot_ttft_total_quality <- function(model_name) {
  qdf <- query_points %>%
    filter(model == model_name) %>%
    arrange(dataset, qaw_ratio)
  selected_ratio_for_model <- model_selected_ratios %>%
    filter(model == model_name) %>%
    pull(selected_ratio) %>%
    .[[1]]
  hdf <- qdf %>% filter(abs(qaw_ratio - selected_ratio_for_model) < 1e-9)

  p <- ggplot(qdf, aes(x = ttft_s, y = total_s)) +
    geom_path(color = "#999999", linewidth = 0.45) +
    geom_point(aes(color = qaw_ratio, size = f1), alpha = 0.9) +
    geom_point(
      data = hdf,
      aes(x = ttft_s, y = total_s),
      inherit.aes = FALSE,
      shape = 21,
      fill = "#FFD166",
      color = "#222222",
      size = 4.0,
      stroke = 0.8
    ) +
    facet_wrap(~ dataset_label, scales = "free", ncol = 1) +
    scale_color_gradient(low = "#3B6EA8", high = "#D95F02") +
    scale_size_continuous(range = c(1.8, 5.0)) +
    labs(
      title = paste(model_name, "TTFT-Total Latency Structure"),
      subtitle = "Point size encodes score; yellow marker highlights the selected setting",
      x = "Average TTFT (s)",
      y = "Average Total Latency (s)",
      color = "Re-compute Ratio",
      size = "Score"
    )

  decorate_common(p) +
    theme(
      strip.text = element_text(face = "bold")
    )
}

save_plot(
  plot_ttft_total_quality("Yi-6B"),
  "fig06_yi6b_ttft_total_quality.png",
  6.8,
  10.5,
  "Yi-6B TTFT-Total Latency Structure",
  "TTFT-total scatter",
  "Yi-6B query-aware curve",
  "Separates first-token delay from total generation latency and marks how quality changes along the curve.",
  "Useful when Chapter 4 discusses TTFT rather than only total latency."
)
#
# save_plot(
#   plot_ttft_total_quality("Qwen2.5-1.5B"),
#   "fig07_qwen25_15b_ttft_total_quality.png",
#   12,
#   7,
#   "Qwen2.5-1.5B TTFT-Total Latency Structure",
#   "TTFT-total scatter",
#   "Qwen2.5-1.5B query-aware curve",
#   "Separates first-token delay from total generation latency and marks how quality changes along the curve.",
#   "Useful when Chapter 4 discusses TTFT rather than only total latency."
# )

heatmap_base <- function(fill_col, fill_label, title, subtitle, low, mid, high, midpoint) {
  p <- ggplot(curve_aug, aes(x = qaw_ratio, y = dataset_label, fill = .data[[fill_col]])) +
    geom_tile(color = "white", linewidth = 0.35) +
    facet_grid(model_label ~ .) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.1), expand = c(0, 0)) +
    scale_fill_gradient2(low = low, mid = mid, high = high, midpoint = midpoint) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "QAW Ratio",
      y = "Dataset",
      fill = fill_label
    )
  decorate_common(p) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

p8 <- heatmap_base(
  "f1_retention_vs_prefill",
  "Score Retention",
  "Quality Retention vs Full Prefill",
  "1.0 means Query-Aware matches Full Prefill score",
  "#B2182B",
  "#F7F7F7",
  "#2166AC",
  1
)
save_plot(
  p8,
  "fig08_f1_retention_heatmap.png",
  10,
  6,
  "Quality Retention vs Full Prefill",
  "Heatmap",
  "Both models, three datasets, all qaw-ratios",
  "Highlights which ratios preserve or exceed full prefill quality.",
  "Useful for selecting qaw-ratio and explaining dataset-specific quality behavior."
)

# p9 <- heatmap_base(
#   "speedup_vs_prefill",
#   "Speedup",
#   "Total Latency Speedup vs Full Prefill",
#   "Values above 1.0 mean Query-Aware is faster than Full Prefill",
#   "#B2182B",
#   "#F7F7F7",
#   "#2166AC",
#   1
# )
# save_plot(
#   p9,
#   "fig09_total_latency_speedup_heatmap.png",
#   10,
#   6,
#   "Total Latency Speedup vs Full Prefill",
#   "Heatmap",
#   "Both models, three datasets, all qaw-ratios",
#   "Shows where query-aware recomputation saves total latency and where it becomes slower.",
#   "Useful for latency analysis and negative-result discussion."
# )
#
# p10 <- ggplot(curve_aug, aes(x = qaw_ratio, y = dataset_label, fill = balanced_score)) +
#   geom_tile(color = "white", linewidth = 0.35) +
#   facet_grid(model_label ~ .) +
#   scale_x_continuous(breaks = seq(0, 1, by = 0.1), expand = c(0, 0)) +
#   scale_fill_gradient(low = "#F7FBFF", high = "#08519C") +
#   labs(
#     title = "Balanced Quality-Latency Score",
#     subtitle = "Score = 0.65 * clipped F1 retention + 0.35 * clipped latency speedup",
#     x = "QAW Ratio",
#     y = "Dataset",
#     fill = "Score"
#   )
# p10 <- decorate_common(p10) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))
# save_plot(
#   p10,
#   "fig10_balanced_score_heatmap.png",
#   10,
#   6,
#   "Balanced Quality-Latency Score",
#   "Heatmap",
#   "Both models, three datasets, all qaw-ratios",
#   "Combines quality retention and latency speedup into a compact ratio-selection view.",
#   "Useful as supporting material for the selected setting."
# )

delta_f1_df <- bind_rows(
  curve_aug %>%
    transmute(
      dataset_label, model_label, qaw_ratio,
      baseline = "vs Full Reuse",
      delta_f1 = f1_delta_vs_reuse
    ),
  curve_aug %>%
    transmute(
      dataset_label, model_label, qaw_ratio,
      baseline = "vs Full Prefill",
      delta_f1 = f1_delta_vs_prefill
    )
) %>%
  mutate(baseline = factor(baseline, levels = c("vs Full Reuse", "vs Full Prefill")))

p11 <- ggplot(delta_f1_df, aes(x = qaw_ratio, y = delta_f1, color = baseline)) +
  geom_hline(yintercept = 0, linewidth = 0.35, color = "#555555") +
  geom_vline(
    data = model_selected_ratios,
    aes(xintercept = selected_ratio),
    linewidth = 0.35,
    linetype = "dotted",
    color = "#4D4D4D"
  ) +
  geom_line(linewidth = 0.75) +
  facet_grid(model_label ~ dataset_label) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
  scale_color_manual(values = c("vs Full Reuse" = "#0072B2", "vs Full Prefill" = "#D55E00")) +
  labs(
    title = "Query-Aware Score Delta against Baselines",
    subtitle = "Positive values indicate Query-Aware is better",
    x = "QAW Ratio",
    y = "Score Delta",
    color = "Reference"
  )
p11 <- decorate_common(p11)
save_plot(
  p11,
  "fig11_f1_delta_against_baselines.png",
  12,
  7,
  "Query-Aware Score Delta against Baselines",
  "Delta line chart",
  "Both models, three datasets, all qaw-ratios",
  "Shows whether query-aware recomputation improves over full reuse and how far it is from full prefill.",
  "Good for explaining quality recovery and degradation cases."
)

# p12 <- ggplot(curve_aug, aes(x = qaw_ratio, y = latency_overhead_pct)) +
#   geom_hline(yintercept = 0, linewidth = 0.35, color = "#555555") +
#   geom_hline(yintercept = 15, linewidth = 0.35, linetype = "dashed", color = "#999999") +
#   geom_vline(
#     data = model_selected_ratios,
#     aes(xintercept = selected_ratio),
#     linewidth = 0.35,
#     linetype = "dotted",
#     color = "#4D4D4D"
#   ) +
#   geom_line(color = method_colors[["Query-Aware"]], linewidth = 0.75) +
#   geom_point(color = method_colors[["Query-Aware"]], size = 1.8) +
#   facet_grid(model_label ~ dataset_label, scales = "free_y") +
#   scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
#   labs(
#     title = "Query-Aware Total Latency Overhead vs Full Prefill",
#     subtitle = "Dashed line is a +15% overhead reference; dotted line marks the selected setting",
#     x = "QAW Ratio",
#     y = "Latency Overhead (%)"
#   )
# p12 <- decorate_common(p12)
# save_plot(
#   p12,
#   "fig12_latency_overhead_vs_prefill.png",
#   12,
#   7,
#   "Query-Aware Total Latency Overhead vs Full Prefill",
#   "Overhead line chart",
#   "Both models, three datasets, all qaw-ratios",
#   "Quantifies the latency cost of increasing recomputation ratio relative to full prefill.",
#   "Useful for justifying the selected ratio and discussing runtime trade-offs."
# )
#
# selected_aug <- curve_aug %>%
#   inner_join(model_selected_ratios %>% select(model, selected_ratio), by = "model") %>%
#   filter(abs(qaw_ratio - selected_ratio) < 1e-9) %>%
#   transmute(
#     dataset_label,
#     model_label,
#     `Score Retention` = f1_retention_vs_prefill,
#     `Latency Ratio` = latency_ratio_vs_prefill,
#     `TTFT Ratio` = ttft_ratio_vs_prefill
#   ) %>%
#   pivot_longer(c(`Score Retention`, `Latency Ratio`, `TTFT Ratio`),
#                names_to = "metric", values_to = "value")
#
# p13 <- ggplot(selected_aug, aes(x = dataset_label, y = value, fill = metric)) +
#   geom_hline(yintercept = 1, linewidth = 0.35, color = "#555555") +
#   geom_col(position = position_dodge(width = 0.72), width = 0.66) +
#   facet_wrap(~ model_label, nrow = 1) +
#   scale_fill_manual(values = c(
#     "Score Retention" = "#009E73",
#     "Latency Ratio" = "#D55E00",
#     "TTFT Ratio" = "#0072B2"
#   )) +
#   labs(
#     title = "Selected Setting Summary",
#     subtitle = "Values are normalized by Full Prefill; 1.0 means equal",
#     x = "Dataset",
#     y = "Normalized Value",
#     fill = "Metric"
#   )
# p13 <- decorate_common(p13) +
#   theme(axis.text.x = element_text(angle = 20, hjust = 1))
# save_plot(
#   p13,
#   "fig13_selected_ratio_normalized_summary.png",
#   10,
#   5,
#   "Selected Setting Normalized Summary",
#   "Grouped bar chart",
#   "Both models at model-specific selected settings",
#   "Summarizes quality retention, total latency ratio, and TTFT ratio against full prefill.",
#   "Compact figure for reporting the chosen settings."
# )

sample_clean <- sample_raw %>%
  filter(
    dataset %in% dataset_levels,
    model %in% model_levels,
    method %in% names(method_colors)
  ) %>%
  mutate(
    dataset_label = factor(unname(dataset_labels[dataset]), levels = unname(dataset_labels)),
    dataset_score_label = factor(unname(dataset_score_labels[dataset]), levels = unname(dataset_score_labels)),
    model_label = factor(model, levels = model_levels),
    method = factor(method, levels = names(method_colors)),
    qaw_ratio_label = if_else(is.na(qaw_ratio), "NA", sprintf("%.2f", qaw_ratio))
  )

# if (nrow(sample_clean) > 0) {
#   sample_long <- sample_clean %>%
#     pivot_longer(c(f1, total_s, ttft_s), names_to = "metric", values_to = "value") %>%
#     mutate(metric_label = factor(unname(metric_labels[metric]), levels = unname(metric_labels))) %>%
#     filter(is.finite(value))
#
#   p14 <- ggplot(sample_long, aes(x = method, y = value, fill = method)) +
#     geom_boxplot(outlier.alpha = 0.35, width = 0.65) +
#     facet_grid(metric_label ~ model_label + dataset_label, scales = "free_y") +
#     scale_fill_manual(values = method_colors) +
#     labs(
#       title = "Available Sample-Level Metric Distributions",
#       subtitle = "Uses fixed-ratio sample_result logs in current outputs/*.output",
#       x = "Method",
#       y = NULL,
#       fill = "Method"
#     )
#   p14 <- decorate_common(p14) +
#     theme(axis.text.x = element_text(angle = 35, hjust = 1))
#   save_plot(
#     p14,
#     "fig14_available_sample_metric_distributions.png",
#     16,
#     9,
#     "Available Sample-Level Metric Distributions",
#     "Boxplot grid",
#     "Fixed-ratio sample_result logs available in current outputs/*.output",
#     "Shows sample-level dispersion of F1, total latency, and TTFT for the three methods.",
#     "Useful as auxiliary evidence for variance and outlier discussion."
#   )
#
#   sample_wide <- sample_clean %>%
#     select(source_file, model_label, dataset_label, qaw_ratio, sample_idx,
#            method_key, f1, total_s, ttft_s, recomputed_tokens) %>%
#     pivot_wider(
#       names_from = method_key,
#       values_from = c(f1, total_s, ttft_s, recomputed_tokens)
#     ) %>%
#     mutate(
#       f1_delta_qaw_vs_prefill = f1_query_aware - f1_full_prefill,
#       latency_delta_qaw_vs_prefill = total_s_query_aware - total_s_full_prefill,
#       ttft_delta_qaw_vs_prefill = ttft_s_query_aware - ttft_s_full_prefill
#     ) %>%
#     filter(is.finite(f1_delta_qaw_vs_prefill), is.finite(latency_delta_qaw_vs_prefill))
#
#   p15 <- ggplot(
#     sample_wide,
#     aes(x = latency_delta_qaw_vs_prefill, y = f1_delta_qaw_vs_prefill)
#   ) +
#     geom_hline(yintercept = 0, linewidth = 0.35, color = "#555555") +
#     geom_vline(xintercept = 0, linewidth = 0.35, color = "#555555") +
#     geom_point(alpha = 0.65, color = method_colors[["Query-Aware"]], size = 2.2) +
#     facet_grid(model_label ~ dataset_label, scales = "free") +
#     labs(
#       title = "Sample-Level Query-Aware Delta vs Full Prefill",
#       subtitle = "Upper-left is better: lower latency and higher score",
#       x = "Total Latency Delta (s)",
#       y = "Score Delta"
#     )
#   p15 <- decorate_common(p15)
#   save_plot(
#     p15,
#     "fig15_sample_delta_vs_full_prefill.png",
#     13,
#     7,
#     "Sample-Level Query-Aware Delta vs Full Prefill",
#     "Sample scatter",
#     "Fixed-ratio sample_result logs available in current outputs/*.output",
#     "Identifies samples where query-aware recomputation gains or loses quality and latency compared with full prefill.",
#     "Useful for case analysis or error analysis subsection."
#   )
#
#   qaw_samples <- sample_clean %>%
#     filter(
#       method_key == "query_aware",
#       is.finite(recomputed_tokens),
#       is.finite(total_s),
#       is.finite(f1)
#     )
#
#   if (nrow(qaw_samples) > 0) {
#     p16 <- ggplot(qaw_samples, aes(x = recomputed_tokens, y = total_s)) +
#       geom_point(aes(color = f1), alpha = 0.75, size = 2.4) +
#       facet_grid(model_label ~ dataset_label, scales = "free") +
#       scale_color_gradient(low = "#FEE8C8", high = "#E34A33") +
#       labs(
#         title = "Recomputed Tokens vs Query-Aware Latency",
#         subtitle = "Available sample_result logs only",
#         x = "Recomputed Tokens",
#         y = "Query-Aware Total Latency (s)",
#         color = "Score"
#       )
#     p16 <- decorate_common(p16)
#     save_plot(
#       p16,
#       "fig16_recomputed_tokens_vs_latency.png",
#       13,
#       7,
#       "Recomputed Tokens vs Query-Aware Latency",
#       "Sample scatter",
#       "Fixed-ratio query-aware sample_result logs available in current outputs/*.output",
#       "Checks whether recomputation workload explains latency variation and whether higher workload aligns with quality changes.",
#       "Useful as auxiliary analysis of runtime mechanism."
#     )
#   }
# }

write.csv(curve_aug, file.path(out_dir, "qaw_curve_metrics.csv"), row.names = FALSE)
write.csv(ratio_scores, file.path(out_dir, "qaw_ratio_scores.csv"), row.names = FALSE)
if (nrow(sample_clean) > 0) {
  write.csv(sample_clean, file.path(out_dir, "qaw_sample_metrics.csv"), row.names = FALSE)
}
write.csv(figure_registry, file.path(out_dir, "figure_inventory.csv"), row.names = FALSE)

selected_score <- curve_aug %>%
  inner_join(model_selected_ratios %>% select(model, selected_ratio), by = "model") %>%
  filter(abs(qaw_ratio - selected_ratio) < 1e-9) %>%
  summarise(
    avg_f1_retention = mean(f1_retention_vs_prefill, na.rm = TRUE),
    avg_speedup_vs_prefill = mean(speedup_vs_prefill, na.rm = TRUE),
    avg_latency_overhead_pct = mean(latency_overhead_pct, na.rm = TRUE),
    avg_query_aware_f1 = mean(query_aware_f1, na.rm = TRUE),
    avg_query_aware_total_s = mean(query_aware_total_s, na.rm = TRUE)
  )

selection_description <- if (is.finite(forced_ratio)) {
  sprintf("主对比图使用的 qaw-ratio 为 `%.2f`。该值由命令行参数 `--qaw-ratio=%.2f` 手动指定。", selected_ratio, forced_ratio)
} else {
  sprintf(
    "主对比图使用按模型分别设置的展示 ratio：`Yi-6B = %.2f`，`Qwen2.5-1.5B = %.2f`。",
    model_selected_ratios$selected_ratio[model_selected_ratios$model == "Yi-6B"],
    model_selected_ratios$selected_ratio[model_selected_ratios$model == "Qwen2.5-1.5B"]
  )
}

md_lines <- c(
  "# Query-Aware 实验可视化图表说明",
  "",
  "说明：图表标题、坐标轴和图例均使用英文，以规避论文绘图阶段中文字体显示异常；本说明文档使用中文记录用途。",
  "",
  sprintf("- 数据来源：`%s`", normalizePath(outputs_dir, mustWork = FALSE)),
  sprintf("- 图表输出目录：`%s`", normalizePath(fig_dir, mustWork = FALSE)),
  sprintf("- 脚本路径：`%s`", normalizePath(file.path(script_dir, basename(script_path_from_args() %||% "visualize_qaw_experiments.R")), mustWork = FALSE)),
  sprintf("- 生成时间：`%s`", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  "",
  "## qaw-ratio 选择",
  "",
  selection_description,
  "",
  sprintf("- 平均质量保持率：`%.4f`", selected_score$avg_f1_retention),
  sprintf("- 平均 total latency speedup vs Full Prefill：`%.4f`", selected_score$avg_speedup_vs_prefill),
  sprintf("- 平均 total latency overhead：`%.2f%%`", selected_score$avg_latency_overhead_pct),
  sprintf("- 平均 Query-Aware 分数：`%.4f`", selected_score$avg_query_aware_f1),
  sprintf("- 平均 Query-Aware total latency：`%.4f s`", selected_score$avg_query_aware_total_s),
  "",
  "## 新绘制图表及作用",
  ""
)

for (i in seq_len(nrow(figure_registry))) {
  item <- figure_registry[i, ]
  md_lines <- c(
    md_lines,
    sprintf("### %02d. `%s`", i, item$file_name),
    "",
    sprintf("- 英文图题：%s", item$title),
    sprintf("- 图表类型：%s", item$chart_type),
    sprintf("- 数据范围：%s", item$data_scope),
    sprintf("- 作用/意义：%s", item$meaning),
    sprintf("- 论文使用建议：%s", item$paper_use),
    ""
  )
}

md_lines <- c(
  md_lines,
  "## 附带数据文件",
  "",
  "- `qaw_curve_metrics.csv`：从 curve run_summary 解析出的宽表，含 score、total_s、TTFT、retention、speedup 等派生指标。",
  "- `qaw_ratio_scores.csv`：每个 qaw-ratio 在 8 个模型-数据集组合上的平均质量-延迟得分。",
  "- `qaw_sample_metrics.csv`：若当前 outputs 根目录存在 sample_result，则保存样本级指标。",
  "- `figure_inventory.csv`：图表清单的机器可读版本。"
)

writeLines(md_lines, con = doc_path, useBytes = TRUE)

message(sprintf(
  "Selected display ratios: %s",
  paste(sprintf("%s=%.2f", model_selected_ratios$model, model_selected_ratios$selected_ratio), collapse = ", ")
))
message(sprintf("Figures written to: %s", normalizePath(fig_dir, mustWork = FALSE)))
message(sprintf("Report written to: %s", normalizePath(doc_path, mustWork = FALSE)))
