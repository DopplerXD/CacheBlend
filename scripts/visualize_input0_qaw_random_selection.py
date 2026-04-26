#!/usr/bin/env python3
"""Save QAW and random selected tokens for inputs/0.json."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Sequence, TextIO, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select tokens from inputs/0.json with QAW and random policies. "
            "This script only uses tokenizer/embedding lookup and does not run "
            "KV cache, prefill, recomputation, or generation."
        ))
    parser.add_argument("--input-file",
                        type=str,
                        default=os.path.join(ROOT_DIR, "inputs", "0.json"))
    parser.add_argument("--model-name",
                        type=str,
                        default="",
                        help="override RuntimeConfig.model_name; default is Yi-6B")
    parser.add_argument("--random-seed",
                        type=int,
                        default=2026,
                        help="base seed for random token selection")
    return parser.parse_args()


def build_output_path() -> str:
    output_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(output_dir, f"input0_qaw_random_tokens_{timestamp}.output")


def write_line(output_file: TextIO, line: str = "") -> None:
    output_file.write(line + "\n")


def ratio_grid() -> List[float]:
    return [round(i * 0.05, 2) for i in range(21)]


def load_input0(path: str) -> Tuple[List[str], str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    chunk_num = int(payload.get("chunk_num", 0))
    if chunk_num <= 0:
        numeric_keys = sorted(
            [key for key in payload.keys() if key.isdigit()],
            key=lambda key: int(key),
        )
    else:
        numeric_keys = [str(i) for i in range(chunk_num)]

    chunks = [str(payload[key]) for key in numeric_keys if key in payload]
    query = str(payload.get("query", ""))
    if not chunks:
        raise ValueError(f"no chunk text found in {path}")
    if not query.strip():
        raise ValueError(f"missing query text in {path}")
    return chunks, query, payload


def encode_chunk_tokens(model_runner: Any, chunk_texts: Sequence[str]) -> List[int]:
    token_ids: List[int] = []
    for text in chunk_texts:
        token_ids.extend(model_runner.encode_no_special(text))
    return token_ids


def compute_qaw_scores(model_runner: Any,
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


def selected_count(token_count: int, ratio: float) -> int:
    if token_count <= 0 or ratio <= 0:
        return 0
    return min(token_count, max(1, int(token_count * ratio)))


def select_qaw_indices(scores: Any, ratio: float) -> Any:
    import torch

    topk_num = selected_count(int(scores.numel()), ratio)
    if topk_num == 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)
    return torch.sort(torch.topk(scores, k=topk_num).indices).values


def select_random_indices(token_count: int,
                          ratio: float,
                          seed: int,
                          device: Any) -> Any:
    import torch

    topk_num = selected_count(token_count, ratio)
    if topk_num == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    indices = torch.randperm(token_count, device=device, generator=generator)[:topk_num]
    return torch.sort(indices).values


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
    return ", ".join(
        clean_token_label(token if token is not None else str(token_id))
        for token, token_id in zip(tokens, selected_ids)
    )


def run(args: argparse.Namespace, output_file: TextIO, output_path: str) -> None:
    from config import RuntimeConfig
    from model.hf_model import HFModelRunner
    from utils.logging_utils import setup_logger

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
    except Exception:
        pass

    cfg = RuntimeConfig()
    if args.model_name.strip():
        cfg.model_name = args.model_name.strip()

    logger = setup_logger("input0_qaw_random_selection", cfg.log_level)
    logger.disabled = True

    chunks, query, _ = load_input0(args.input_file)
    model_runner = HFModelRunner(cfg.model_name, cfg.device, cfg.model_dtype, logger)

    chunk_token_ids = encode_chunk_tokens(model_runner, chunks)
    query_token_ids = model_runner.encode_no_special(query)
    scores = compute_qaw_scores(model_runner, chunk_token_ids, query_token_ids)
    ratios = ratio_grid()

    write_line(output_file, f"output_path: {output_path}")
    write_line(output_file, f"input_file: {args.input_file}")
    write_line(output_file, f"model: {cfg.model_name}")
    write_line(output_file, f"chunk_count: {len(chunks)}")
    write_line(output_file, f"chunk_token_count: {len(chunk_token_ids)}")
    write_line(output_file, f"query_token_count: {len(query_token_ids)}")
    write_line(output_file, "ratio_step: 0.05")
    write_line(output_file,
               "ratio_values: " + ", ".join(f"{ratio:.2f}" for ratio in ratios))
    write_line(output_file)

    for ratio in ratios:
        qaw_indices = select_qaw_indices(scores, ratio)
        random_indices = select_random_indices(
            token_count=len(chunk_token_ids),
            ratio=ratio,
            seed=args.random_seed + int(round(ratio * 100)),
            device=model_runner.device,
        )
        write_line(output_file, f"ratio: {ratio:.2f}")
        write_line(output_file, f"qaw_selected_count: {int(qaw_indices.numel())}")
        write_line(output_file, f"qaw_selected_tokens: {selected_token_text(model_runner, chunk_token_ids, qaw_indices)}")
        write_line(output_file, f"random_selected_count: {int(random_indices.numel())}")
        write_line(output_file, f"random_selected_tokens: {selected_token_text(model_runner, chunk_token_ids, random_indices)}")
        write_line(output_file)


def main() -> None:
    args = parse_args()
    output_path = build_output_path()
    with open(output_path, "w", encoding="utf-8") as output_file:
        with contextlib.redirect_stdout(output_file), contextlib.redirect_stderr(output_file):
            run(args, output_file, output_path)


if __name__ == "__main__":
    main()
