#!/usr/bin/env python3
"""Print query-aware selected tokens without running cache recomputation."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, List, Sequence

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

EXAMPLE_DIR = os.path.join(ROOT_DIR, "example")
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

SUPPORTED_DATASETS = ("cmrc", "musique", "samsum", "wikimqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize query-aware token selection only. This script does not "
            "run KV cache reuse, recomputation, prefill, or generation."
        ))
    parser.add_argument("--dataset",
                        choices=SUPPORTED_DATASETS,
                        required=True)
    parser.add_argument("--count", type=int, default=1, help="number of samples")
    parser.add_argument("--qaw-ratio",
                        type=float,
                        default=0.7,
                        help="top-k ratio over chunk tokens")
    parser.add_argument("--model-name",
                        type=str,
                        default="",
                        help="override RuntimeConfig.model_name; default is Yi-6B")
    return parser.parse_args()


def encode_chunk_tokens(model_runner: Any, chunk_texts: Sequence[str]) -> List[int]:
    token_ids: List[int] = []
    for text in chunk_texts:
        token_ids.extend(model_runner.encode_no_special(text))
    return token_ids


def compute_query_aware_scores(model_runner: Any,
                               chunk_token_ids: List[int],
                               query_token_ids: List[int]) -> Any:
    import torch

    if not chunk_token_ids:
        return torch.zeros(0, device=model_runner.device)
    if not query_token_ids:
        return torch.zeros(len(chunk_token_ids), device=model_runner.device)

    cand_emb = model_runner.lookup_token_embeddings(chunk_token_ids)
    query_emb = model_runner.lookup_token_embeddings(query_token_ids)
    cand_emb = torch.nn.functional.normalize(cand_emb, dim=-1)
    query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
    sim = cand_emb @ query_emb.transpose(0, 1)
    return sim.max(dim=1).values


def select_chunk_indices(scores: Any, recomp_ratio: float) -> Any:
    import torch

    chunk_len = int(scores.numel())
    if chunk_len == 0 or recomp_ratio <= 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)

    topk_num = min(chunk_len, max(1, int(chunk_len * recomp_ratio)))
    selected = torch.topk(scores, k=topk_num).indices
    return torch.sort(selected).values


def clean_token_label(token: str) -> str:
    token = token.replace("\r", "\\r").replace("\n", "\\n")
    token = token.replace("▁", " ").replace("Ġ", " ")
    return token if token.strip() else "<ws>"


def selected_token_text(model_runner: Any,
                        chunk_token_ids: List[int],
                        selected_indices: Any) -> str:
    selected_ids = [
        chunk_token_ids[int(idx)] for idx in selected_indices.detach().cpu().tolist()
    ]
    if not selected_ids:
        return ""

    tokens = model_runner.tokenizer.convert_ids_to_tokens(selected_ids)
    if isinstance(tokens, str):
        tokens = [tokens]
    labels = [
        clean_token_label(token if token is not None else str(token_id))
        for token, token_id in zip(tokens, selected_ids)
    ]
    return ", ".join(labels)


def main() -> None:
    args = parse_args()
    if args.count < 0:
        raise ValueError("--count must be >= 0")

    from config import RuntimeConfig
    from blend_curve_common import build_sample_parts, load_dataset
    from model.hf_model import HFModelRunner
    from utils.logging_utils import setup_logger

    spec, eval_dataset = load_dataset(args.dataset)
    sample_limit = min(args.count, len(eval_dataset))

    cfg = RuntimeConfig()
    if args.model_name.strip():
        cfg.model_name = args.model_name.strip()

    logger = setup_logger("visualize_selected_tokens", cfg.log_level)
    logger.disabled = True
    model_runner = HFModelRunner(cfg.model_name, cfg.device, cfg.model_dtype, logger)

    for sample_idx, example in enumerate(eval_dataset[:sample_limit], start=1):
        doc_prompts, q_prompt, query_text = build_sample_parts(example, spec)
        chunk_token_ids = encode_chunk_tokens(model_runner, doc_prompts)

        query_source = query_text.strip() or q_prompt.strip()
        query_token_ids = (model_runner.encode_no_special(query_source)
                           if query_source else [])
        scores = compute_query_aware_scores(
            model_runner=model_runner,
            chunk_token_ids=chunk_token_ids,
            query_token_ids=query_token_ids,
        )
        selected_indices = select_chunk_indices(scores, args.qaw_ratio)
        print(
            f"sample {sample_idx} selected_tokens: "
            f"{selected_token_text(model_runner, chunk_token_ids, selected_indices)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
