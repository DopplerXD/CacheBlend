#!/usr/bin/env python3
"""Generate random chunk/query and intra-chunk similarity heatmaps."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency hint
    raise ImportError("This script requires matplotlib. Install requirements.txt first.") from exc


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(ROOT_DIR, "example")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

from blend_curve_common import build_sample_parts, load_dataset
from config import RuntimeConfig
from model.hf_model import HFModelRunner
from utils.logging_utils import setup_logger


DATASETS = ("musique", "wikimqa", "samsum")


@dataclass(frozen=True)
class TokenWindow:
    token_ids: List[int]
    start: int
    end: int
    total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample one example and one chunk from musique, wikimqa, "
            "and samsum, then export chunk-query and chunk-token similarity heatmaps."
        )
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Override RuntimeConfig.model_name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            ROOT_DIR,
            "graduation_project",
            "analyse",
            "random_chunk_similarity",
        ),
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=128,
        help=(
            "Deprecated compatibility option. Chunk-side heatmaps always use "
            "the full selected chunk."
        ),
    )
    parser.add_argument(
        "--max-query-tokens",
        type=int,
        default=96,
        help="Maximum visible query tokens. Use 0 to disable truncation.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_chunk_tokens < 0:
        raise SystemExit("--max-chunk-tokens must be >= 0")
    if args.max_query_tokens < 0:
        raise SystemExit("--max-query-tokens must be >= 0")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be > 0")


def random_window(token_ids: Sequence[int], max_tokens: int, rng: random.Random) -> TokenWindow:
    total = len(token_ids)
    if total == 0:
        raise ValueError("cannot visualize an empty token sequence")
    if max_tokens == 0 or total <= max_tokens:
        start = 0
        end = total
    else:
        start = rng.randint(0, total - max_tokens)
        end = start + max_tokens
    return TokenWindow(token_ids=list(token_ids[start:end]), start=start, end=end, total=total)


def clean_token_label(token: str, max_chars: int = 18) -> str:
    token = token.replace("\r", "\\r").replace("\n", "\\n")
    token = token.replace("▁", " ").replace("Ġ", " ")
    token = token if token.strip() else "<ws>"
    if len(token) > max_chars:
        return token[: max_chars - 1] + "..."
    return token


def decode_token_labels(tokenizer: Any, token_ids: Sequence[int], offset: int = 0) -> List[str]:
    raw_tokens = tokenizer.convert_ids_to_tokens([int(token_id) for token_id in token_ids])
    if isinstance(raw_tokens, str):
        raw_tokens = [raw_tokens]
    labels: List[str] = []
    for idx, (token, token_id) in enumerate(zip(raw_tokens, token_ids)):
        token_text = token if token is not None else str(token_id)
        labels.append(f"{offset + idx}:{clean_token_label(token_text)}")
    return labels


def tick_positions(label_count: int, max_ticks: int = 48) -> List[int]:
    if label_count <= max_ticks:
        return list(range(label_count))
    stride = max(1, (label_count + max_ticks - 1) // max_ticks)
    positions = list(range(0, label_count, stride))
    if positions[-1] != label_count - 1:
        positions.append(label_count - 1)
    return positions


@torch.inference_mode()
def cosine_similarity_matrix(
    model_runner: HFModelRunner,
    row_token_ids: Sequence[int],
    col_token_ids: Sequence[int],
) -> Any:
    row_emb = model_runner.lookup_token_embeddings([int(token_id) for token_id in row_token_ids])
    col_emb = model_runner.lookup_token_embeddings([int(token_id) for token_id in col_token_ids])
    row_emb = torch.nn.functional.normalize(row_emb.float(), dim=-1)
    col_emb = torch.nn.functional.normalize(col_emb.float(), dim=-1)
    return (row_emb @ col_emb.transpose(0, 1)).detach().cpu().numpy()


@torch.inference_mode()
def cosine_similarity_to_mean_query(
    model_runner: HFModelRunner,
    chunk_token_ids: Sequence[int],
    query_token_ids: Sequence[int],
) -> Any:
    chunk_emb = model_runner.lookup_token_embeddings(
        [int(token_id) for token_id in chunk_token_ids]
    )
    query_emb = model_runner.lookup_token_embeddings(
        [int(token_id) for token_id in query_token_ids]
    )
    chunk_emb = torch.nn.functional.normalize(chunk_emb.float(), dim=-1)
    query_vec = query_emb.float().mean(dim=0, keepdim=True)
    query_vec = torch.nn.functional.normalize(query_vec, dim=-1)
    return (chunk_emb @ query_vec.transpose(0, 1)).detach().cpu().numpy()


def plot_heatmap(
    matrix: Any,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    dpi: int,
) -> None:
    width = max(6.0, min(12.0, 0.12 * len(x_labels) + 4.0))
    height = max(5.0, min(12.0, 0.10 * len(y_labels) + 3.0))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", interpolation="nearest")

    x_ticks = tick_positions(len(x_labels))
    y_ticks = tick_positions(len(y_labels))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([x_labels[i] for i in x_ticks], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([y_labels[i] for i in y_ticks], fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Cosine Similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def normalize_answers(raw_answers: Any) -> List[str]:
    if raw_answers is None:
        return []
    if not isinstance(raw_answers, list):
        return [str(raw_answers)]
    answers: List[str] = []
    for item in raw_answers:
        if isinstance(item, str):
            answers.append(item)
        elif isinstance(item, list):
            answers.extend(str(value) for value in item if value is not None)
        elif item is not None:
            answers.append(str(item))
    return answers


def build_output_prefix(output_dir: str, dataset: str, sample_idx: int, chunk_idx: int) -> str:
    return os.path.join(output_dir, f"{dataset}_sample{sample_idx:03d}_chunk{chunk_idx:02d}")


def visualize_dataset(
    dataset: str,
    rng: random.Random,
    model_runner: HFModelRunner,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    spec, eval_dataset = load_dataset(dataset)
    if not eval_dataset:
        raise ValueError(f"{dataset} dataset is empty")

    sample_zero_idx = rng.randrange(len(eval_dataset))
    sample_idx = sample_zero_idx + 1
    example = eval_dataset[sample_zero_idx]
    doc_prompts, q_prompt, query_text = build_sample_parts(example, spec)
    if not doc_prompts:
        raise ValueError(f"{dataset} sample {sample_idx} has no chunks")

    chunk_zero_idx = rng.randrange(len(doc_prompts))
    chunk_idx = chunk_zero_idx + 1
    chunk_text = doc_prompts[chunk_zero_idx]
    query_source = query_text.strip() or q_prompt.strip()
    if not query_source:
        raise ValueError(f"{dataset} sample {sample_idx} has no query text")

    chunk_token_ids = model_runner.encode_no_special(chunk_text)
    query_token_ids = model_runner.encode_no_special(query_source)
    query_window = random_window(query_token_ids, args.max_query_tokens, rng)

    output_prefix = build_output_prefix(args.output_dir, dataset, sample_idx, chunk_idx)
    query_similarity_path = f"{output_prefix}_query_similarity.png"
    query_mean_similarity_path = f"{output_prefix}_query_mean_similarity.png"
    chunk_similarity_path = f"{output_prefix}_chunk_token_similarity.png"

    chunk_labels = decode_token_labels(
        model_runner.tokenizer,
        chunk_token_ids,
        offset=0,
    )
    query_labels = decode_token_labels(
        model_runner.tokenizer,
        query_window.token_ids,
        offset=query_window.start,
    )

    query_matrix = cosine_similarity_matrix(
        model_runner=model_runner,
        row_token_ids=chunk_token_ids,
        col_token_ids=query_window.token_ids,
    )
    chunk_matrix = cosine_similarity_matrix(
        model_runner=model_runner,
        row_token_ids=chunk_token_ids,
        col_token_ids=chunk_token_ids,
    )
    query_mean_matrix = cosine_similarity_to_mean_query(
        model_runner=model_runner,
        chunk_token_ids=chunk_token_ids,
        query_token_ids=query_window.token_ids,
    )

    plot_heatmap(
        matrix=query_matrix,
        x_labels=query_labels,
        y_labels=chunk_labels,
        title=f"{dataset} Sample {sample_idx} Chunk {chunk_idx}: Chunk Tokens x Query Tokens",
        xlabel="Query Tokens",
        ylabel="Chunk Tokens",
        output_path=query_similarity_path,
        dpi=args.dpi,
    )
    plot_heatmap(
        matrix=chunk_matrix,
        x_labels=chunk_labels,
        y_labels=chunk_labels,
        title=f"{dataset} Sample {sample_idx} Chunk {chunk_idx}: Chunk Token Similarity",
        xlabel="Chunk Tokens",
        ylabel="Chunk Tokens",
        output_path=chunk_similarity_path,
        dpi=args.dpi,
    )
    plot_heatmap(
        matrix=query_mean_matrix,
        x_labels=["mean(query)"],
        y_labels=chunk_labels,
        title=f"{dataset} Sample {sample_idx} Chunk {chunk_idx}: Chunk Tokens x Mean Query Vector",
        xlabel="Mean-Pooled Query Vector",
        ylabel="Chunk Tokens",
        output_path=query_mean_similarity_path,
        dpi=args.dpi,
    )

    return {
        "dataset": dataset,
        "sample_idx": sample_idx,
        "chunk_idx": chunk_idx,
        "sample_count": len(eval_dataset),
        "chunk_count": len(doc_prompts),
        "question": str(example.get("question", "")),
        "answers": normalize_answers(example.get("answers", [])),
        "uses_full_chunk_tokens": True,
        "chunk_token_total": len(chunk_token_ids),
        "query_token_total": query_window.total,
        "query_window": {
            "start": query_window.start,
            "end": query_window.end,
            "token_count": len(query_window.token_ids),
            "is_truncated": len(query_window.token_ids) < query_window.total,
        },
        "query_similarity": {
            "path": query_similarity_path,
            "matrix_shape": list(query_matrix.shape),
        },
        "query_mean_similarity": {
            "path": query_mean_similarity_path,
            "matrix_shape": list(query_mean_matrix.shape),
            "pooled_query_token_count": len(query_window.token_ids),
        },
        "chunk_token_similarity": {
            "path": chunk_similarity_path,
            "matrix_shape": list(chunk_matrix.shape),
        },
    }


def write_manifest(output_dir: str, args: argparse.Namespace, entries: List[Dict[str, Any]]) -> str:
    manifest_path = os.path.join(output_dir, "random_chunk_similarity_manifest.json")
    payload = {
        "seed": args.seed,
        "model_name": args.model_name,
        "uses_full_chunk_tokens": True,
        "deprecated_max_chunk_tokens": args.max_chunk_tokens,
        "max_query_tokens": args.max_query_tokens,
        "datasets": entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = RuntimeConfig()
    if args.model_name.strip():
        cfg.model_name = args.model_name.strip()
    args.model_name = cfg.model_name

    logger = setup_logger("random_chunk_similarity", cfg.log_level)
    logger.disabled = True
    model_runner = HFModelRunner(cfg.model_name, cfg.device, cfg.model_dtype, logger)

    rng = random.Random(args.seed)
    entries = [
        visualize_dataset(dataset, rng=rng, model_runner=model_runner, args=args)
        for dataset in DATASETS
    ]
    manifest_path = write_manifest(args.output_dir, args, entries)
    print(f"saved manifest: {manifest_path}")
    for entry in entries:
        print(f"saved figure: {entry['query_similarity']['path']}")
        print(f"saved figure: {entry['query_mean_similarity']['path']}")
        print(f"saved figure: {entry['chunk_token_similarity']['path']}")


if __name__ == "__main__":
    main()
