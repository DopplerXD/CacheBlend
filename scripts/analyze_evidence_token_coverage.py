#!/usr/bin/env python3
"""Measure whether selectors cover answer-string evidence tokens."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qaw_analysis_utils import (
    EMBEDDING_VARIANTS,
    HIDDEN_VARIANTS,
    answer_evidence_indices,
    build_model_runner,
    chunk_head_indices,
    compute_embedding_scores,
    compute_hidden_scores,
    compute_query_attention_scores,
    evidence_metrics,
    iter_tokenized_samples,
    make_run_output_dir,
    parse_datasets,
    parse_ratios,
    random_indices,
    stable_seed,
    topk_indices,
    write_csv,
    write_json,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_METHODS = (
    "random",
    "chunk_head",
    "embedding_max",
    "embedding_mean_query",
    "embedding_last_query",
    "query_attention",
)
SUPPORTED_METHODS = DEFAULT_METHODS + HIDDEN_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use answer-string spans and a small token window as weak evidence, "
            "then compare selector coverage without running generation."
        )
    )
    parser.add_argument("--datasets", default="musique,wikimqa")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--model-name", default="/root/models/Yi-6B")
    parser.add_argument(
        "--ratios",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--evidence-window", type=int, default=5)
    parser.add_argument("--attention-layer", type=int, default=-1)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--skip-long-samples",
        action="store_true",
        help=(
            "skip samples longer than --max-prompt-tokens; by default long "
            "samples are truncated for lightweight analysis"
        ),
    )
    parser.add_argument("--bar-ratio", type=float, default=0.6)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            ROOT_DIR,
            "graduation_project",
            "analyse",
            "qaw_theory_support",
            "evidence_token_coverage",
        ),
    )
    parser.add_argument(
        "--no-run-subdir",
        action="store_true",
        help="write directly into --output-dir; default creates a timestamped run subdirectory",
    )
    return parser.parse_args()


def parse_methods(raw_value: str) -> List[str]:
    methods = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    invalid = [method for method in methods if method not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"unsupported methods: {', '.join(invalid)}")
    return methods or list(DEFAULT_METHODS)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_rows(rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, float], Dict[str, List[float]]] = defaultdict(
        lambda: {"recall": [], "hit": []}
    )
    for row in rows:
        key = (row["dataset"], row["method"], float(row["ratio"]))
        if row["evidence_recall"] != "":
            grouped[key]["recall"].append(float(row["evidence_recall"]))
        if row["evidence_hit"] != "":
            grouped[key]["hit"].append(float(row["evidence_hit"]))
    out = []
    for (dataset, method, ratio), values in sorted(grouped.items()):
        out.append({
            "dataset": dataset,
            "method": method,
            "ratio": ratio,
            "mean_evidence_recall": mean(values["recall"]),
            "mean_evidence_hit": mean(values["hit"]),
            "sample_count": len(values["recall"]),
        })
    return out


def plot_recall_curve(aggregate: Sequence[Dict], output_dir: str,
                      dataset_order: Sequence[str]) -> str:
    datasets = list(dataset_order)
    methods = sorted({row["method"] for row in aggregate})
    fig, axes = plt.subplots(
        1,
        max(1, len(datasets)),
        figsize=(6.2 * max(1, len(datasets)), 4.8),
        squeeze=False,
    )
    for ax, dataset in zip(axes[0], datasets):
        for method in methods:
            rows = [
                row for row in aggregate
                if row["dataset"] == dataset and row["method"] == method
            ]
            if not rows:
                continue
            rows = sorted(rows, key=lambda item: float(item["ratio"]))
            ax.plot(
                [float(row["ratio"]) for row in rows],
                [float(row["mean_evidence_recall"]) for row in rows],
                marker="o",
                linewidth=1.6,
                label=method,
            )
        ax.set_title(f"{dataset}: Evidence Recall")
        ax.set_xlabel("Top-K Ratio")
        ax.set_ylabel("Mean Evidence Recall")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path = os.path.join(output_dir, "evidence_recall_curve.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_dataset_summary(sample_rows: Sequence[Dict],
                          skipped_rows: Sequence[Dict],
                          dataset_order: Sequence[str]) -> List[Dict]:
    summary = []
    for dataset in dataset_order:
        processed = [row for row in sample_rows if row["dataset"] == dataset]
        skipped = [row for row in skipped_rows if row["dataset"] == dataset]
        truncated = [row for row in processed if row.get("was_truncated")]
        no_evidence = [
            row for row in skipped
            if row.get("reason") == "answer evidence not found in chunk tokens"
        ]
        summary.append({
            "dataset": dataset,
            "processed_samples": len(processed),
            "skipped_or_truncated_records": len(skipped),
            "truncated_samples": len(truncated),
            "no_evidence_samples": len(no_evidence),
        })
    return summary


def plot_bar_at_ratio(aggregate: Sequence[Dict], output_dir: str, ratio: float) -> str:
    rows = [row for row in aggregate if abs(float(row["ratio"]) - ratio) < 1e-9]
    if not rows:
        return ""
    datasets = sorted({row["dataset"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    x_labels = [f"{dataset}\n{method}" for dataset in datasets for method in methods]
    values = []
    for dataset in datasets:
        for method in methods:
            match = [
                row for row in rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            values.append(float(match[0]["mean_evidence_recall"]) if match else 0.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(x_labels)), 4.8))
    ax.bar(range(len(x_labels)), values, color="#4C78A8")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_title(f"Evidence Recall at Ratio {ratio:.2f}")
    ax.set_ylabel("Mean Evidence Recall")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, f"evidence_recall_ratio_{ratio:.2f}.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def select_for_method(method: str, ratio: float, length: int, scores_by_method: Dict[str, object],
                      dataset: str, sample_id: str) -> List[int]:
    if method == "random":
        return random_indices(length, ratio, stable_seed(dataset, sample_id, ratio, method))
    if method == "chunk_head":
        return chunk_head_indices(length, ratio)
    return topk_indices(scores_by_method[method], ratio)


def main() -> None:
    args = parse_args()
    datasets = parse_datasets(args.datasets)
    ratios = parse_ratios(args.ratios)
    methods = parse_methods(args.methods)
    base_output_dir = args.output_dir
    output_dir = make_run_output_dir(
        base_output_dir=args.output_dir,
        run_name="evidence",
        datasets=datasets,
        no_run_subdir=args.no_run_subdir,
    )

    needs_attention = "query_attention" in methods
    attn_impl = "eager" if needs_attention else "auto"
    cfg, model_runner = build_model_runner(args.model_name, attn_implementation=attn_impl)
    coverage_rows: List[Dict] = []
    sample_rows: List[Dict] = []
    skipped_rows: List[Dict] = []

    for _, sample in iter_tokenized_samples(
        model_runner=model_runner,
        datasets=datasets,
        count=args.count,
        max_prompt_tokens=args.max_prompt_tokens,
        skipped_rows=skipped_rows,
        truncate_long_samples=not args.skip_long_samples,
    ):
        evidence = answer_evidence_indices(
            model_runner=model_runner,
            sample=sample,
            window_tokens=args.evidence_window,
        )
        if not evidence:
            skipped_rows.append({
                "dataset": sample.dataset,
                "sample_idx": sample.sample_idx,
                "sample_id": sample.sample_id,
                "reason": "answer evidence not found in chunk tokens",
            })
            continue

        query_token_ids = (
            model_runner.encode_no_special(sample.query_source)
            if sample.query_source else []
        )
        scores_by_method: Dict[str, object] = {}
        try:
            for method in methods:
                if method in EMBEDDING_VARIANTS:
                    scores_by_method[method] = compute_embedding_scores(
                        model_runner=model_runner,
                        chunk_token_ids=sample.chunk_token_ids,
                        query_source_token_ids=query_token_ids,
                        variant=method,
                    )
                elif method == "query_attention":
                    scores_by_method[method] = compute_query_attention_scores(
                        model_runner=model_runner,
                        sample=sample,
                        attention_layer=args.attention_layer,
                    )
                elif method in HIDDEN_VARIANTS:
                    scores_by_method[method] = compute_hidden_scores(
                        model_runner=model_runner,
                        sample=sample,
                        variant=method,
                    )
        except RuntimeError as exc:
            skipped_rows.append({
                "dataset": sample.dataset,
                "sample_idx": sample.sample_idx,
                "sample_id": sample.sample_id,
                "reason": str(exc),
            })
            continue

        sample_rows.append({
            "dataset": sample.dataset,
            "sample_idx": sample.sample_idx,
            "sample_id": sample.sample_id,
            "prompt_tokens": len(sample.prompt_token_ids),
            "original_prompt_tokens": sample.original_prompt_tokens,
            "was_truncated": sample.was_truncated,
            "chunk_tokens": len(sample.chunk_token_ids),
            "evidence_tokens": len(evidence),
            "answers": " | ".join(sample.answers),
        })
        for ratio in ratios:
            for method in methods:
                selected = select_for_method(
                    method=method,
                    ratio=ratio,
                    length=len(sample.chunk_token_ids),
                    scores_by_method=scores_by_method,
                    dataset=sample.dataset,
                    sample_id=sample.sample_id,
                )
                recall, hit = evidence_metrics(selected, evidence)
                coverage_rows.append({
                    "dataset": sample.dataset,
                    "sample_idx": sample.sample_idx,
                    "sample_id": sample.sample_id,
                    "method": method,
                    "ratio": ratio,
                    "evidence_recall": "" if recall is None else recall,
                    "evidence_hit": "" if hit is None else hit,
                    "selected_tokens": len(selected),
                    "evidence_tokens": len(evidence),
                })

    aggregate = aggregate_rows(coverage_rows)
    coverage_csv = os.path.join(output_dir, "evidence_coverage_by_sample.csv")
    aggregate_csv = os.path.join(output_dir, "evidence_coverage_summary.csv")
    sample_csv = os.path.join(output_dir, "evidence_samples.csv")
    skipped_csv = os.path.join(output_dir, "skipped_samples.csv")
    dataset_summary = build_dataset_summary(sample_rows, skipped_rows, datasets)
    dataset_summary_csv = os.path.join(output_dir, "dataset_summary.csv")
    write_csv(coverage_csv, coverage_rows)
    write_csv(aggregate_csv, aggregate)
    write_csv(sample_csv, sample_rows)
    write_csv(skipped_csv, skipped_rows)
    write_csv(dataset_summary_csv, dataset_summary)

    figure_paths = []
    if aggregate:
        figure_paths.append(plot_recall_curve(aggregate, output_dir, datasets))
        bar_path = plot_bar_at_ratio(aggregate, output_dir, args.bar_ratio)
        if bar_path:
            figure_paths.append(bar_path)

    manifest_path = os.path.join(output_dir, "evidence_coverage_manifest.json")
    write_json(manifest_path, {
        "model": cfg.model_name,
        "datasets": datasets,
        "base_output_dir": base_output_dir,
        "output_dir": output_dir,
        "count_per_dataset": args.count,
        "methods": methods,
        "ratios": ratios,
        "evidence_window": args.evidence_window,
        "attention_layer": args.attention_layer,
        "max_prompt_tokens": args.max_prompt_tokens,
        "truncate_long_samples": not args.skip_long_samples,
        "processed_samples": len(sample_rows),
        "skipped_samples": len(skipped_rows),
        "dataset_summary": dataset_summary,
        "coverage_csv": coverage_csv,
        "aggregate_csv": aggregate_csv,
        "sample_csv": sample_csv,
        "skipped_csv": skipped_csv,
        "dataset_summary_csv": dataset_summary_csv,
        "figures": figure_paths,
    })
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
