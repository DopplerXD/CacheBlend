#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(jsonlite)))

script_file_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_dir <- if (length(script_file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[[1L]])))
} else {
  getwd()
}
repo_root <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)

default_input <- "/Users/doppler/documents/github/cmrc2018/squad-style-data/cmrc2018_dev.json"
default_output <- file.path(repo_root, "inputs", "cmrc_s.json")
default_samples <- 50L
default_seed <- 2026L

args <- commandArgs(trailingOnly = TRUE)
input_json <- default_input
output_json <- default_output
max_samples <- default_samples
seed <- default_seed

if (length(args) > 0L) {
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--input-json" && i + 1L <= length(args)) {
      input_json <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--output-json" && i + 1L <= length(args)) {
      output_json <- args[[i + 1L]]
      i <- i + 2L
    } else if (key == "--max-samples" && i + 1L <= length(args)) {
      max_samples <- as.integer(args[[i + 1L]])
      i <- i + 2L
    } else if (key == "--seed" && i + 1L <= length(args)) {
      seed <- as.integer(args[[i + 1L]])
      i <- i + 2L
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }
}

if (!file.exists(input_json)) {
  stop(sprintf("Input JSON not found: %s", input_json))
}

unique_keep_order <- function(values) {
  seen <- new.env(parent = emptyenv())
  out <- character(0)
  for (value in values) {
    value <- trimws(as.character(value))
    if (!nzchar(value)) {
      next
    }
    if (!exists(value, envir = seen, inherits = FALSE)) {
      assign(value, TRUE, envir = seen)
      out <- c(out, value)
    }
  }
  out
}

cmrc_obj <- fromJSON(input_json, simplifyVector = FALSE)
records <- list()

for (article in cmrc_obj$data) {
  title <- if (!is.null(article$title)) article$title else ""
  for (paragraph in article$paragraphs) {
    context <- if (!is.null(paragraph$context)) paragraph$context else ""
    if (!nzchar(trimws(context))) {
      next
    }
    for (qa in paragraph$qas) {
      question <- if (!is.null(qa$question)) trimws(qa$question) else ""
      if (!nzchar(question)) {
        next
      }
      answers_raw <- if (!is.null(qa$answers)) qa$answers else list()
      answers <- unique_keep_order(vapply(
        answers_raw,
        function(answer_item) {
          if (is.null(answer_item$text)) "" else as.character(answer_item$text)
        },
        FUN.VALUE = character(1)
      ))
      if (length(answers) == 0L) {
        next
      }

      records[[length(records) + 1L]] <- list(
        ctxs = list(list(
          title = title,
          text = context
        )),
        question = question,
        answers = unname(as.list(answers))
      )
    }
  }
}

if (length(records) == 0L) {
  stop("No valid CMRC samples found after preprocessing.")
}

set.seed(seed)
sample_count <- min(max_samples, length(records))
selected_indices <- sample.int(length(records), sample_count, replace = FALSE)
selected_records <- records[selected_indices]

output_dir <- dirname(output_json)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

write_json(
  selected_records,
  path = output_json,
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null"
)

cat(sprintf(
  "Done. input=%s output=%s samples=%d seed=%d\n",
  input_json,
  output_json,
  length(selected_records),
  seed
))
