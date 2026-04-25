#!/usr/bin/env python3
"""Save query-aware selected tokens without running cache recomputation."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, List, Sequence

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

EXAMPLE_DIR = os.path.join(ROOT_DIR, "example")
if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

SUPPORTED_DATASETS = ("cmrc", "musique", "samsum", "wikimqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save query-aware token selection only. This script does not "
            "run KV cache reuse, recomputation, prefill, or generation."
        ))
    parser.add_argument("--dataset",
                        choices=SUPPORTED_DATASETS,
                        required=True)
    parser.add_argument("--count", type=int, default=1, help="number of samples")
    parser.add_argument("--qaw-ratio",
                        type=ratio_value,
                        default=0.6,
                        help="default top-k ratio over chunk tokens")
    parser.add_argument(
        "--ratio-min",
        type=ratio_value,
        default=None,
        help="minimum qaw ratio to evaluate; defaults to --qaw-ratio",
    )
    parser.add_argument(
        "--ratio-max",
        type=ratio_value,
        default=None,
        help="maximum qaw ratio to evaluate; defaults to --qaw-ratio",
    )
    parser.add_argument("--model-name",
                        type=str,
                        default="",
                        help="override RuntimeConfig.model_name; default is Yi-6B")
    return parser.parse_args()


def ratio_value(raw_value: str) -> float:
    value = float(raw_value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def build_output_path() -> str:
    output_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(output_dir, f"selected_tokens_{timestamp}.output")


def write_line(line: str, output_file: Any) -> None:
    output_file.write(line + "\n")


def ratio_grid(ratio_min: float, ratio_max: float, step: float = 0.05) -> List[float]:
    if ratio_min > ratio_max:
        raise SystemExit("--ratio-min must be <= --ratio-max")

    values: List[float] = []
    current = round(ratio_min, 2)
    end = round(ratio_max, 2)
    while current <= end + 1e-9:
        values.append(round(current, 2))
        current = round(current + step, 10)
    if not values:
        values = [round(ratio_min, 2)]
    return values


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
        raise SystemExit("--count must be >= 0")
    ratio_min = args.qaw_ratio if args.ratio_min is None else args.ratio_min
    ratio_max = args.qaw_ratio if args.ratio_max is None else args.ratio_max
    ratios = ratio_grid(ratio_min, ratio_max)

    from config import RuntimeConfig
    from blend_curve_common import build_sample_parts, load_dataset
    from model.hf_model import HFModelRunner
    from utils.logging_utils import setup_logger
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except Exception:
        pass

    spec, eval_dataset = load_dataset(args.dataset)
    sample_limit = min(args.count, len(eval_dataset))

    cfg = RuntimeConfig()
    if args.model_name.strip():
        cfg.model_name = args.model_name.strip()

    logger = setup_logger("visualize_selected_tokens", cfg.log_level)
    logger.disabled = True
    model_runner = HFModelRunner(cfg.model_name, cfg.device, cfg.model_dtype, logger)

    output_path = build_output_path()
    with open(output_path, "w", encoding="utf-8") as output_file:
        write_line(f"output_path: {output_path}", output_file)
        write_line(f"dataset: {args.dataset}", output_file)
        write_line(f"count: {sample_limit}", output_file)
        write_line(f"qaw_ratio_default: {args.qaw_ratio}", output_file)
        write_line(f"ratio_min: {ratio_min}", output_file)
        write_line(f"ratio_max: {ratio_max}", output_file)
        write_line("ratio_step: 0.05", output_file)
        write_line(
            "ratio_values: " + ", ".join(f"{ratio:.2f}" for ratio in ratios),
            output_file,
        )
        write_line(f"model: {cfg.model_name}", output_file)

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
            for ratio in ratios:
                selected_indices = select_chunk_indices(scores, ratio)
                selected_text = selected_token_text(
                    model_runner=model_runner,
                    chunk_token_ids=chunk_token_ids,
                    selected_indices=selected_indices,
                )
                write_line(
                    f"qaw_ratio {ratio:.2f} sample {sample_idx} selected_tokens: {selected_text}",
                    output_file,
                )

        output_file.flush()


if __name__ == "__main__":
    main()
