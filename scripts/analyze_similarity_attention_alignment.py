#!/usr/bin/env python3
"""Compare query-token embedding similarity with query-to-context attention."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qaw_analysis_utils import (
    EMBEDDING_VARIANTS,
    build_model_runner,
    compute_embedding_scores,
    compute_query_attention_scores,
    iter_tokenized_samples,
    make_run_output_dir,
    parse_datasets,
    parse_ratios,
    pearson_corr,
    spearman_corr,
    topk_overlap,
    write_csv,
    write_json,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether lightweight query-token embedding similarity "
            "is aligned with model query-to-context attention."
        )
    )
    parser.add_argument("--datasets", default="musique,wikimqa,samsum")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--model-name", default="/root/models/Yi-6B")
    parser.add_argument("--similarity-variant",
                        choices=EMBEDDING_VARIANTS,
                        default="embedding_max")
    parser.add_argument(
        "--ratios",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--attention-layer", type=int, default=-1)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="0 means no prompt-length cap; set a positive value to cap analysis length",
    )
    parser.add_argument(
        "--skip-long-samples",
        action="store_true",
        help=(
            "compatibility flag; when --max-prompt-tokens is positive, long "
            "samples are skipped unless --truncate-long-samples is set"
        ),
    )
    parser.add_argument(
        "--truncate-long-samples",
        action="store_true",
        help=(
            "when --max-prompt-tokens is positive, truncate chunk tokens "
            "instead of skipping long samples"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            ROOT_DIR,
            "graduation_project",
            "analyse",
            "qaw_theory_support",
            "similarity_attention_alignment",
        ),
    )
    parser.add_argument(
        "--no-run-subdir",
        action="store_true",
        help="write directly into --output-dir; default creates a timestamped run subdirectory",
    )
    return parser.parse_args()


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_overlap_groups(overlap_rows: List[Dict]) -> Dict[Tuple[str, float], List[float]]:
    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for row in overlap_rows:
        if row["topk_overlap"] != "":
            grouped[(row["dataset"], float(row["ratio"]))].append(
                float(row["topk_overlap"])
            )
    return grouped


def plot_overlap(overlap_rows: List[Dict], output_dir: str,
                 dataset_order: List[str]) -> str:
    grouped = build_overlap_groups(overlap_rows)
    datasets = [dataset for dataset in dataset_order
                if any(key[0] == dataset for key in grouped)]
    ratios = sorted({key[1] for key in grouped})

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for dataset in datasets:
        y_values = [mean(grouped.get((dataset, ratio), [])) for ratio in ratios]
        ax.plot(ratios, y_values, marker="o", linewidth=1.8, label=dataset)
    ax.set_title("Similarity-Attention Top-K Overlap")
    ax.set_xlabel("Top-K Ratio")
    ax.set_ylabel("Mean Top-K Overlap")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "similarity_attention_topk_overlap.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_average_overlap(overlap_rows: List[Dict], output_dir: str,
                         dataset_order: List[str]) -> str:
    grouped = build_overlap_groups(overlap_rows)
    ratios = sorted({key[1] for key in grouped})
    mean_values = []
    for ratio in ratios:
        dataset_means = []
        for dataset in dataset_order:
            values = grouped.get((dataset, ratio), [])
            if values:
                dataset_means.append(mean(values))
        mean_values.append(mean(dataset_means))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(ratios, mean_values, marker="o", linewidth=2.0, color="#222222")
    ax.set_title("Average Similarity-Attention Top-K Overlap")
    ax.set_xlabel("Top-K Ratio")
    ax.set_ylabel("Mean Top-K Overlap across Datasets")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, "similarity_attention_topk_overlap_average.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spearman_by_dataset(sample_rows: List[Dict],
                             output_dir: str,
                             dataset_order: List[str]) -> List[str]:
    paths = []
    for dataset in dataset_order:
        values = [
            float(row["spearman"]) for row in sample_rows
            if row["dataset"] == dataset and row["spearman"] != ""
        ]
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        if values:
            ax.boxplot([values], labels=[dataset], showmeans=True)
        else:
            ax.set_xticks([1])
            ax.set_xticklabels([dataset])
            ax.text(
                0.5,
                0.5,
                "No usable samples",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(f"{dataset}: Similarity-Attention Spearman")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Spearman")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(
            output_dir,
            f"similarity_attention_spearman_boxplot_{dataset}.png",
        )
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def build_dataset_summary(sample_rows: List[Dict], skipped_rows: List[Dict],
                          dataset_order: List[str]) -> List[Dict]:
    summary = []
    for dataset in dataset_order:
        processed = [row for row in sample_rows if row["dataset"] == dataset]
        skipped = [row for row in skipped_rows if row["dataset"] == dataset]
        truncated = [row for row in processed if row.get("was_truncated")]
        summary.append({
            "dataset": dataset,
            "processed_samples": len(processed),
            "skipped_or_truncated_records": len(skipped),
            "truncated_samples": len(truncated),
        })
    return summary


def main() -> None:
    args = parse_args()
    datasets = parse_datasets(args.datasets)
    ratios = parse_ratios(args.ratios)
    base_output_dir = args.output_dir
    output_dir = make_run_output_dir(
        base_output_dir=args.output_dir,
        run_name="alignment",
        datasets=datasets,
        no_run_subdir=args.no_run_subdir,
    )

    # Attention weights are needed here; eager is more reliable than SDPA/FlashAttention.
    cfg, model_runner = build_model_runner(args.model_name, attn_implementation="eager")
    sample_rows: List[Dict] = []
    overlap_rows: List[Dict] = []
    skipped_rows: List[Dict] = []
    truncate_long_samples = (
        args.max_prompt_tokens > 0
        and args.truncate_long_samples
        and not args.skip_long_samples
    )

    for _, sample in iter_tokenized_samples(
        model_runner=model_runner,
        datasets=datasets,
        count=args.count,
        max_prompt_tokens=args.max_prompt_tokens,
        skipped_rows=skipped_rows,
        truncate_long_samples=truncate_long_samples,
    ):
        query_token_ids = (
            model_runner.encode_no_special(sample.query_source)
            if sample.query_source else []
        )
        try:
            similarity_scores = compute_embedding_scores(
                model_runner=model_runner,
                chunk_token_ids=sample.chunk_token_ids,
                query_source_token_ids=query_token_ids,
                variant=args.similarity_variant,
            )
            attention_scores = compute_query_attention_scores(
                model_runner=model_runner,
                sample=sample,
                attention_layer=args.attention_layer,
            )
        except (RuntimeError, ValueError) as exc:
            skipped_rows.append({
                "dataset": sample.dataset,
                "sample_idx": sample.sample_idx,
                "sample_id": sample.sample_id,
                "reason": str(exc),
            })
            continue

        pearson = pearson_corr(similarity_scores, attention_scores)
        spearman = spearman_corr(similarity_scores, attention_scores)
        sample_rows.append({
            "dataset": sample.dataset,
            "sample_idx": sample.sample_idx,
            "sample_id": sample.sample_id,
            "prompt_tokens": len(sample.prompt_token_ids),
            "original_prompt_tokens": sample.original_prompt_tokens,
            "was_truncated": sample.was_truncated,
            "chunk_tokens": len(sample.chunk_token_ids),
            "query_tokens": len(query_token_ids),
            "similarity_variant": args.similarity_variant,
            "pearson": "" if pearson is None else pearson,
            "spearman": "" if spearman is None else spearman,
        })
        for ratio in ratios:
            overlap = topk_overlap(similarity_scores, attention_scores, ratio)
            overlap_rows.append({
                "dataset": sample.dataset,
                "sample_idx": sample.sample_idx,
                "sample_id": sample.sample_id,
                "similarity_variant": args.similarity_variant,
                "ratio": ratio,
                "topk_overlap": "" if overlap is None else overlap,
            })

    sample_csv = os.path.join(output_dir, "sample_similarity_attention.csv")
    overlap_csv = os.path.join(output_dir, "topk_overlap_by_sample.csv")
    skipped_csv = os.path.join(output_dir, "skipped_samples.csv")
    dataset_summary = build_dataset_summary(sample_rows, skipped_rows, datasets)
    dataset_summary_csv = os.path.join(output_dir, "dataset_summary.csv")
    write_csv(sample_csv, sample_rows)
    write_csv(overlap_csv, overlap_rows)
    write_csv(skipped_csv, skipped_rows)
    write_csv(dataset_summary_csv, dataset_summary)

    figure_paths = []
    if overlap_rows:
        figure_paths.append(plot_overlap(overlap_rows, output_dir, datasets))
        figure_paths.append(plot_average_overlap(overlap_rows, output_dir, datasets))
    if sample_rows:
        figure_paths.extend(plot_spearman_by_dataset(sample_rows, output_dir, datasets))

    summary = {
        "model": cfg.model_name,
        "datasets": datasets,
        "base_output_dir": base_output_dir,
        "output_dir": output_dir,
        "count_per_dataset": args.count,
        "similarity_variant": args.similarity_variant,
        "attention_layer": args.attention_layer,
        "max_prompt_tokens": args.max_prompt_tokens,
        "truncate_long_samples": truncate_long_samples,
        "processed_samples": len(sample_rows),
        "skipped_samples": len(skipped_rows),
        "dataset_summary": dataset_summary,
        "sample_csv": sample_csv,
        "overlap_csv": overlap_csv,
        "skipped_csv": skipped_csv,
        "dataset_summary_csv": dataset_summary_csv,
        "figures": figure_paths,
    }
    manifest = os.path.join(output_dir, "alignment_manifest.json")
    write_json(manifest, summary)
    print(f"saved manifest: {manifest}")


if __name__ == "__main__":
    main()
