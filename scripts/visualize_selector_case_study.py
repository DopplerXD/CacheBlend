#!/usr/bin/env python3
"""Generate a selector case study with token-level highlights."""

from __future__ import annotations

import argparse
import html
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qaw_analysis_utils import (
    TARGET_DATASETS,
    answer_evidence_indices,
    build_model_runner,
    chunk_head_indices,
    clear_cuda_cache,
    compute_embedding_scores,
    compute_query_attention_scores,
    decode_token_labels,
    ensure_dir,
    evidence_metrics,
    iter_tokenized_samples,
    random_indices,
    stable_seed,
    topk_indices,
    write_json,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD_ORDER = ("evidence", "embedding_max", "query_attention", "chunk_head", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically find a clear sample where query-aware selection "
            "covers answer evidence better than simple baselines, then export "
            "HTML and PNG visualizations."
        )
    )
    parser.add_argument("--dataset", choices=TARGET_DATASETS, default="wikimqa")
    parser.add_argument("--model-name", default="/root/models/Yi-6B")
    parser.add_argument("--scan-limit", type=int, default=80)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--evidence-window", type=int, default=5)
    parser.add_argument("--attention-layer", type=int, default=-1)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4096,
        help="positive values cap analysis length; 0 means no prompt-length cap",
    )
    parser.add_argument(
        "--skip-long-samples",
        action="store_true",
        help=(
            "when --max-prompt-tokens is positive, skip long samples instead "
            "of truncating chunk tokens"
        ),
    )
    parser.add_argument("--window-tokens", type=int, default=180)
    parser.add_argument(
        "--allow-attention-reference",
        action="store_true",
        help=(
            "If no answer-string evidence is found, use high-attention tokens "
            "as a weak reference. Useful only for SAMSum-style qualitative figures."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            ROOT_DIR,
            "graduation_project",
            "analyse",
            "qaw_theory_support",
            "selector_case_study",
        ),
    )
    return parser.parse_args()


def metric(selected: List[int], evidence: List[int]) -> float:
    recall, hit = evidence_metrics(selected, evidence)
    if recall is None or hit is None:
        return 0.0
    return recall + 0.25 * hit


def choose_window(evidence: List[int], selected: Dict[str, List[int]], total: int,
                  window_tokens: int) -> Tuple[int, int]:
    if total <= window_tokens:
        return 0, total
    anchors = evidence or selected.get("embedding_max", []) or [0]
    center = anchors[len(anchors) // 2]
    start = max(0, center - window_tokens // 2)
    end = min(total, start + window_tokens)
    start = max(0, end - window_tokens)
    return start, end


def build_case_html(
    output_path: str,
    sample,
    token_labels: List[str],
    evidence: List[int],
    selected: Dict[str, List[int]],
    window: Tuple[int, int],
    evidence_source: str,
) -> None:
    start, end = window
    evidence_set = set(evidence)
    selected_sets = {method: set(indices) for method, indices in selected.items()}
    spans = []
    for local_idx in range(start, end):
        classes = []
        if local_idx in evidence_set:
            classes.append("evidence")
        for method in ("embedding_max", "query_attention", "chunk_head", "random"):
            if local_idx in selected_sets.get(method, set()):
                classes.append(method)
        title = f"idx={local_idx}; " + ",".join(classes)
        spans.append(
            f'<span class="tok {" ".join(classes)}" title="{html.escape(title)}">'
            f"{html.escape(token_labels[local_idx])}</span>"
        )

    body = "\n".join([
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;line-height:1.65;margin:24px;}",
        ".meta{max-width:980px;margin-bottom:16px;}",
        ".tokens{max-width:1120px;border:1px solid #ddd;padding:14px;}",
        ".tok{display:inline-block;margin:2px;padding:2px 4px;border:1px solid #ccc;border-radius:3px;}",
        ".evidence{background:#fff1a8;}",
        ".embedding_max{border:2px solid #1b9e77;}",
        ".query_attention{box-shadow:inset 0 -3px 0 #7570b3;}",
        ".chunk_head{outline:2px dotted #d95f02;}",
        ".random{color:#b2182b;}",
        ".legend span{margin-right:12px;}",
        "</style></head><body>",
        "<div class='meta'>",
        f"<h2>{html.escape(sample.dataset)} sample {sample.sample_idx}</h2>",
        f"<p><b>Question / query:</b> {html.escape(sample.query_source)}</p>",
        f"<p><b>Answers:</b> {html.escape(' | '.join(sample.answers))}</p>",
        f"<p><b>Evidence source:</b> {html.escape(evidence_source)}</p>",
        "<p class='legend'>",
        "<span class='tok evidence'>evidence</span>",
        "<span class='tok embedding_max'>embedding_max</span>",
        "<span class='tok query_attention'>query_attention</span>",
        "<span class='tok chunk_head'>chunk_head</span>",
        "<span class='tok random'>random</span>",
        "</p></div>",
        "<div class='tokens'>",
        "".join(spans),
        "</div></body></html>",
    ])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)


def plot_case_tracks(
    output_path: str,
    evidence: List[int],
    selected: Dict[str, List[int]],
    window: Tuple[int, int],
) -> None:
    start, end = window
    width = max(1, end - start)
    rows = []
    labels = []
    evidence_set = set(evidence)
    for method in METHOD_ORDER:
        labels.append(method)
        active = evidence_set if method == "evidence" else set(selected.get(method, []))
        rows.append([1 if idx in active else 0 for idx in range(start, end)])

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.imshow(rows, aspect="auto", interpolation="nearest", cmap="Greens")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    tick_stride = max(1, width // 12)
    tick_positions = list(range(0, width, tick_stride))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(start + pos) for pos in tick_positions], rotation=30)
    ax.set_xlabel("Chunk Token Index")
    ax.set_title("Selector Mask Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    cfg, model_runner = build_model_runner(args.model_name, attn_implementation="eager")
    truncate_long_samples = args.max_prompt_tokens > 0 and not args.skip_long_samples

    best_case = None
    best_score = -1.0
    skipped = []
    for _, sample in iter_tokenized_samples(
        model_runner=model_runner,
        datasets=[args.dataset],
        count=args.scan_limit,
        max_prompt_tokens=args.max_prompt_tokens,
        skipped_rows=skipped,
        truncate_long_samples=truncate_long_samples,
    ):
        query_token_ids = (
            model_runner.encode_no_special(sample.query_source)
            if sample.query_source else []
        )
        try:
            embedding_scores = compute_embedding_scores(
                model_runner=model_runner,
                chunk_token_ids=sample.chunk_token_ids,
                query_source_token_ids=query_token_ids,
                variant="embedding_max",
            )
            attention_scores = compute_query_attention_scores(
                model_runner=model_runner,
                sample=sample,
                attention_layer=args.attention_layer,
            )
        except (RuntimeError, ValueError) as exc:
            clear_cuda_cache()
            skipped.append({
                "sample_idx": sample.sample_idx,
                "sample_id": sample.sample_id,
                "prompt_tokens": len(sample.prompt_token_ids),
                "chunk_tokens": len(sample.chunk_token_ids),
                "reason": str(exc),
            })
            continue

        evidence = answer_evidence_indices(
            model_runner=model_runner,
            sample=sample,
            window_tokens=args.evidence_window,
        )
        evidence_source = "answer_string_window"
        if not evidence and args.allow_attention_reference:
            evidence = topk_indices(attention_scores, args.ratio)
            evidence_source = "query_attention_reference"
        if not evidence:
            skipped.append({
                "sample_idx": sample.sample_idx,
                "sample_id": sample.sample_id,
                "reason": "no answer evidence found",
            })
            continue

        selected = {
            "embedding_max": topk_indices(embedding_scores, args.ratio),
            "query_attention": topk_indices(attention_scores, args.ratio),
            "chunk_head": chunk_head_indices(len(sample.chunk_token_ids), args.ratio),
            "random": random_indices(
                len(sample.chunk_token_ids),
                args.ratio,
                stable_seed(sample.dataset, sample.sample_id, args.ratio, "case_random"),
            ),
        }
        baseline = max(metric(selected["chunk_head"], evidence),
                       metric(selected["random"], evidence))
        score = metric(selected["embedding_max"], evidence) - baseline
        if score > best_score:
            best_score = score
            best_case = {
                "sample": sample,
                "evidence": evidence,
                "evidence_source": evidence_source,
                "selected": selected,
                "score": score,
            }

    if best_case is None:
        raise RuntimeError("No usable case found. Try a larger --scan-limit.")

    sample = best_case["sample"]
    evidence = best_case["evidence"]
    selected = best_case["selected"]
    token_labels = decode_token_labels(model_runner.tokenizer, sample.chunk_token_ids)
    window = choose_window(
        evidence=evidence,
        selected=selected,
        total=len(sample.chunk_token_ids),
        window_tokens=args.window_tokens,
    )
    prefix = os.path.join(
        args.output_dir,
        f"{sample.dataset}_sample{sample.sample_idx:03d}_ratio{args.ratio:.2f}",
    )
    html_path = f"{prefix}_tokens.html"
    png_path = f"{prefix}_selector_tracks.png"
    manifest_path = f"{prefix}_manifest.json"

    build_case_html(
        output_path=html_path,
        sample=sample,
        token_labels=token_labels,
        evidence=evidence,
        selected=selected,
        window=window,
        evidence_source=best_case["evidence_source"],
    )
    plot_case_tracks(
        output_path=png_path,
        evidence=evidence,
        selected=selected,
        window=window,
    )

    method_metrics = {}
    for method, indices in selected.items():
        recall, hit = evidence_metrics(indices, evidence)
        method_metrics[method] = {
            "evidence_recall": recall,
            "evidence_hit": hit,
            "selected_tokens": len(indices),
        }

    write_json(manifest_path, {
        "model": cfg.model_name,
        "dataset": sample.dataset,
        "sample_idx": sample.sample_idx,
        "sample_id": sample.sample_id,
        "ratio": args.ratio,
        "max_prompt_tokens": args.max_prompt_tokens,
        "truncate_long_samples": truncate_long_samples,
        "question": sample.query_source,
        "answers": sample.answers,
        "prompt_tokens": len(sample.prompt_token_ids),
        "chunk_tokens": len(sample.chunk_token_ids),
        "evidence_source": best_case["evidence_source"],
        "evidence_tokens": len(evidence),
        "window": {"start": window[0], "end": window[1]},
        "method_metrics": method_metrics,
        "html": html_path,
        "selector_tracks_png": png_path,
        "skipped_samples": skipped,
    })
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
