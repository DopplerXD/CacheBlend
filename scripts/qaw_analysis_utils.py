"""Shared helpers for lightweight query-aware analysis scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(ROOT_DIR, "example")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if EXAMPLE_DIR not in sys.path:
    sys.path.append(EXAMPLE_DIR)

from blend_curve_common import DATASET_SPECS, build_sample_parts, load_dataset
from config import RuntimeConfig
from model.hf_model import HFModelRunner
from utils.logging_utils import setup_logger

TARGET_DATASETS = ("musique", "wikimqa", "samsum")
EMBEDDING_VARIANTS = (
    "embedding_max",
    "embedding_mean_query",
    "embedding_last_query",
)
HIDDEN_VARIANTS = ("hidden_last", "hidden_multi_mean")


@dataclass
class TokenizedSample:
    dataset: str
    sample_idx: int
    sample_id: str
    example: Dict[str, Any]
    answers: List[str]
    doc_prompts: List[str]
    q_prompt: str
    query_text: str
    query_source: str
    prefix_token_ids: List[int]
    chunk_token_ids: List[int]
    suffix_token_ids: List[int]
    prompt_token_ids: List[int]
    chunk_global_positions: List[int]
    query_global_positions: List[int]


def parse_datasets(raw_value: str) -> List[str]:
    datasets = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    invalid = [item for item in datasets if item not in TARGET_DATASETS]
    if invalid:
        raise ValueError(f"unsupported datasets: {', '.join(invalid)}")
    return datasets or list(TARGET_DATASETS)


def parse_ratios(raw_value: str) -> List[float]:
    ratios = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    for ratio in ratios:
        if ratio < 0 or ratio > 1:
            raise ValueError("ratios must be within [0, 1]")
    return ratios or [0.05, 0.1, 0.2, 0.3, 0.5]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_run_output_dir(base_output_dir: str,
                        run_name: str,
                        datasets: Sequence[str],
                        no_run_subdir: bool = False) -> str:
    """Create a per-run output directory to avoid overwriting prior datasets."""
    if no_run_subdir:
        ensure_dir(base_output_dir)
        return base_output_dir
    dataset_slug = "-".join(datasets) if datasets else "datasets"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(base_output_dir, f"{timestamp}_{run_name}_{dataset_slug}")
    ensure_dir(run_dir)
    return run_dir


def build_model_runner(model_name: str, attn_implementation: str) -> Tuple[RuntimeConfig, HFModelRunner]:
    cfg = RuntimeConfig()
    if model_name.strip():
        cfg.model_name = model_name.strip()
    cfg.attn_implementation = attn_implementation
    logger = setup_logger("qaw_analysis", cfg.log_level)
    logger.disabled = True
    runner = HFModelRunner(
        cfg.model_name,
        cfg.device,
        cfg.model_dtype,
        logger,
        cfg.attn_implementation,
    )
    return cfg, runner


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def find_subsequence_positions(full_ids: Sequence[int], sub_ids: Sequence[int]) -> List[int]:
    if not full_ids or not sub_ids or len(sub_ids) > len(full_ids):
        return []
    last_start = -1
    limit = len(full_ids) - len(sub_ids) + 1
    sub = list(sub_ids)
    for start in range(limit):
        if list(full_ids[start:start + len(sub)]) == sub:
            last_start = start
    if last_start < 0:
        return []
    return list(range(last_start, last_start + len(sub)))


def find_all_subsequences(full_ids: Sequence[int], sub_ids: Sequence[int]) -> List[Tuple[int, int]]:
    if not full_ids or not sub_ids or len(sub_ids) > len(full_ids):
        return []
    spans: List[Tuple[int, int]] = []
    sub = list(sub_ids)
    limit = len(full_ids) - len(sub) + 1
    for start in range(limit):
        if list(full_ids[start:start + len(sub)]) == sub:
            spans.append((start, start + len(sub)))
    return spans


def normalize_answer_variants(answer: str) -> List[str]:
    cleaned = " ".join(str(answer).strip().split())
    if not cleaned:
        return []
    lowered = cleaned.lower()
    no_punc = re.sub(r"[^\w\s]", " ", cleaned)
    no_punc = " ".join(no_punc.split())
    variants = [cleaned, f" {cleaned}", lowered, f" {lowered}"]
    if no_punc:
        variants.extend([no_punc, f" {no_punc}", no_punc.lower(), f" {no_punc.lower()}"])
    deduped: List[str] = []
    seen = set()
    for value in variants:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def build_tokenized_sample(
    model_runner: HFModelRunner,
    dataset: str,
    sample_idx: int,
    example: Dict[str, Any],
    spec: Dict[str, Any],
) -> TokenizedSample:
    doc_prompts, q_prompt, query_text = build_sample_parts(example, spec)
    prefix_token_ids = (
        model_runner.encode(spec["prefix_prompt"]) if spec.get("prefix_prompt") else []
    )
    chunk_token_ids: List[int] = []
    for chunk_text in doc_prompts:
        chunk_token_ids.extend(model_runner.encode_no_special(chunk_text))
    suffix_token_ids = model_runner.encode_no_special(q_prompt) if q_prompt else []
    prompt_token_ids = prefix_token_ids + chunk_token_ids + suffix_token_ids
    chunk_start = len(prefix_token_ids)
    chunk_global_positions = list(range(chunk_start, chunk_start + len(chunk_token_ids)))

    answer_fn = spec["answer_fn"]
    answers = answer_fn(example.get("answers", []))
    query_source = (
        (query_text or "").strip()
        or (q_prompt or "").strip()
        or str(example.get("question", "")).strip()
        or " ".join(answers).strip()
    )
    query_token_ids = (
        model_runner.encode_no_special(query_source) if query_source else []
    )
    query_positions = find_subsequence_positions(prompt_token_ids, query_token_ids)
    if not query_positions and suffix_token_ids:
        query_count = min(len(query_token_ids), len(suffix_token_ids))
        suffix_start = len(prefix_token_ids) + len(chunk_token_ids)
        query_positions = list(
            range(
                suffix_start + max(0, len(suffix_token_ids) - query_count),
                suffix_start + len(suffix_token_ids),
            )
        )

    sample_id = str(example.get("_id") or example.get("id") or sample_idx)
    return TokenizedSample(
        dataset=dataset,
        sample_idx=sample_idx,
        sample_id=sample_id,
        example=example,
        answers=answers,
        doc_prompts=doc_prompts,
        q_prompt=q_prompt,
        query_text=query_text,
        query_source=query_source,
        prefix_token_ids=prefix_token_ids,
        chunk_token_ids=chunk_token_ids,
        suffix_token_ids=suffix_token_ids,
        prompt_token_ids=prompt_token_ids,
        chunk_global_positions=chunk_global_positions,
        query_global_positions=query_positions,
    )


@torch.inference_mode()
def compute_embedding_scores(
    model_runner: HFModelRunner,
    chunk_token_ids: Sequence[int],
    query_source_token_ids: Sequence[int],
    variant: str = "embedding_max",
) -> torch.Tensor:
    if not chunk_token_ids:
        return torch.zeros(0)
    if not query_source_token_ids:
        return torch.zeros(len(chunk_token_ids))
    variant = (variant or "embedding_max").lower()
    if variant not in EMBEDDING_VARIANTS:
        raise ValueError(f"unsupported embedding variant: {variant}")

    cand_emb = model_runner.lookup_token_embeddings([int(x) for x in chunk_token_ids])
    query_emb = model_runner.lookup_token_embeddings([int(x) for x in query_source_token_ids])
    cand_emb = torch.nn.functional.normalize(cand_emb.float(), dim=-1)
    query_emb = torch.nn.functional.normalize(query_emb.float(), dim=-1)
    if variant == "embedding_mean_query":
        query_vec = torch.nn.functional.normalize(query_emb.mean(dim=0, keepdim=True), dim=-1)
        scores = cand_emb @ query_vec.transpose(0, 1)
        return scores.squeeze(1).detach().cpu()
    if variant == "embedding_last_query":
        scores = cand_emb @ query_emb[-1:].transpose(0, 1)
        return scores.squeeze(1).detach().cpu()
    scores = cand_emb @ query_emb.transpose(0, 1)
    return scores.max(dim=1).values.detach().cpu()


@torch.inference_mode()
def compute_query_attention_scores(
    model_runner: HFModelRunner,
    sample: TokenizedSample,
    attention_layer: int,
) -> torch.Tensor:
    if not sample.chunk_global_positions or not sample.query_global_positions:
        return torch.zeros(len(sample.chunk_token_ids))
    input_ids = torch.tensor(
        [sample.prompt_token_ids],
        dtype=torch.long,
        device=model_runner.device,
    )
    attention_mask = torch.ones(
        (1, len(sample.prompt_token_ids)),
        dtype=torch.long,
        device=model_runner.device,
    )
    outputs = model_runner.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )
    if not getattr(outputs, "attentions", None):
        raise RuntimeError("model did not return attentions; use eager attention")
    attn = outputs.attentions[attention_layer][0]  # [heads, seq, seq]
    query_pos = torch.tensor(sample.query_global_positions, dtype=torch.long, device=attn.device)
    chunk_pos = torch.tensor(sample.chunk_global_positions, dtype=torch.long, device=attn.device)
    scores = attn.index_select(1, query_pos).index_select(2, chunk_pos)
    return scores.mean(dim=(0, 1)).detach().float().cpu()


@torch.inference_mode()
def compute_hidden_scores(
    model_runner: HFModelRunner,
    sample: TokenizedSample,
    variant: str,
) -> torch.Tensor:
    if not sample.chunk_global_positions or not sample.query_global_positions:
        return torch.zeros(len(sample.chunk_token_ids))
    input_ids = torch.tensor(
        [sample.prompt_token_ids],
        dtype=torch.long,
        device=model_runner.device,
    )
    attention_mask = torch.ones(
        (1, len(sample.prompt_token_ids)),
        dtype=torch.long,
        device=model_runner.device,
    )
    outputs = model_runner.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("model did not return hidden states")
    layers = [hidden_states[-1]]
    if variant == "hidden_multi_mean":
        layers = list(hidden_states[-4:])

    chunk_pos = torch.tensor(sample.chunk_global_positions, dtype=torch.long, device=model_runner.device)
    query_pos = torch.tensor(sample.query_global_positions, dtype=torch.long, device=model_runner.device)
    layer_scores = []
    for hidden in layers:
        hidden = hidden[0].float()
        chunk_hidden = hidden.index_select(0, chunk_pos)
        query_hidden = hidden.index_select(0, query_pos)
        chunk_hidden = torch.nn.functional.normalize(chunk_hidden, dim=-1)
        query_hidden = torch.nn.functional.normalize(query_hidden, dim=-1)
        layer_scores.append((chunk_hidden @ query_hidden.transpose(0, 1)).max(dim=1).values)
    return torch.stack(layer_scores, dim=0).mean(dim=0).detach().cpu()


def rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().cpu()
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(len(values), dtype=torch.float32)
    return ranks


def pearson_corr(left: torch.Tensor, right: torch.Tensor) -> Optional[float]:
    if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
        return None
    x = left.detach().float().cpu()
    y = right.detach().float().cpu()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) <= 1e-12:
        return None
    return float((x * y).sum() / denom)


def spearman_corr(left: torch.Tensor, right: torch.Tensor) -> Optional[float]:
    return pearson_corr(rankdata(left), rankdata(right))


def topk_indices(scores: torch.Tensor, ratio: float) -> List[int]:
    count = int(scores.numel())
    if count == 0 or ratio <= 0:
        return []
    k = min(count, max(1, int(count * ratio)))
    return [int(i) for i in torch.topk(scores, k=k).indices.tolist()]


def topk_overlap(left_scores: torch.Tensor, right_scores: torch.Tensor, ratio: float) -> Optional[float]:
    left = set(topk_indices(left_scores, ratio))
    right = set(topk_indices(right_scores, ratio))
    if not left or not right:
        return None
    return len(left & right) / min(len(left), len(right))


def random_indices(length: int, ratio: float, seed: int) -> List[int]:
    if length == 0 or ratio <= 0:
        return []
    k = min(length, max(1, int(length * ratio)))
    rng = random.Random(seed)
    return sorted(rng.sample(range(length), k=k))


def chunk_head_indices(length: int, ratio: float) -> List[int]:
    if length == 0 or ratio <= 0:
        return []
    k = min(length, max(1, int(length * ratio)))
    return list(range(k))


def answer_evidence_indices(
    model_runner: HFModelRunner,
    sample: TokenizedSample,
    window_tokens: int,
) -> List[int]:
    evidence = set()
    for answer in sample.answers:
        for variant in normalize_answer_variants(answer):
            token_ids = model_runner.encode_no_special(variant)
            if not token_ids:
                continue
            for start, end in find_all_subsequences(sample.chunk_token_ids, token_ids):
                lo = max(0, start - window_tokens)
                hi = min(len(sample.chunk_token_ids), end + window_tokens)
                evidence.update(range(lo, hi))
    return sorted(evidence)


def evidence_metrics(selected_indices: Iterable[int], evidence_indices: Sequence[int]) -> Tuple[Optional[float], Optional[float]]:
    evidence = set(int(i) for i in evidence_indices)
    if not evidence:
        return None, None
    selected = set(int(i) for i in selected_indices)
    overlap = selected & evidence
    return len(overlap) / len(evidence), 1.0 if overlap else 0.0


def decode_token_labels(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    raw_tokens = tokenizer.convert_ids_to_tokens([int(token_id) for token_id in token_ids])
    if isinstance(raw_tokens, str):
        raw_tokens = [raw_tokens]
    labels = []
    for token, token_id in zip(raw_tokens, token_ids):
        label = token if token is not None else str(token_id)
        label = label.replace("\r", "\\r").replace("\n", "\\n")
        label = label.replace("▁", " ").replace("Ġ", " ")
        labels.append(label if label.strip() else "<ws>")
    return labels


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def iter_tokenized_samples(
    model_runner: HFModelRunner,
    datasets: Sequence[str],
    count: int,
    max_prompt_tokens: int,
    skipped_rows: Optional[List[Dict[str, Any]]] = None,
):
    for dataset in datasets:
        spec, examples = load_dataset(dataset)
        processed = 0
        for sample_idx, example in enumerate(examples, start=1):
            if processed >= count:
                break
            sample = build_tokenized_sample(
                model_runner=model_runner,
                dataset=dataset,
                sample_idx=sample_idx,
                example=example,
                spec=spec,
            )
            if max_prompt_tokens > 0 and len(sample.prompt_token_ids) > max_prompt_tokens:
                if skipped_rows is not None:
                    skipped_rows.append({
                        "dataset": dataset,
                        "sample_idx": sample_idx,
                        "sample_id": sample.sample_id,
                        "prompt_tokens": len(sample.prompt_token_ids),
                        "max_prompt_tokens": max_prompt_tokens,
                        "reason": "prompt exceeds max_prompt_tokens",
                    })
                continue
            processed += 1
            yield spec, sample
