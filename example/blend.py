"""小规模 chunk-cache 演示：full_prefill / full_reuse / query_aware。"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from typing import List, Optional

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cache.kv_cache import KVCacheManager
from config import RuntimeConfig
from engine.inference_engine import InferenceEngine
from model.hf_model import HFModelRunner
from schema.types import GenerateRequest
from utils.experiment_output import ExperimentOutputWriter, utc8_now_str
from utils.logging_utils import setup_logger


PREFIX_PROMPT = (
    "Answer the question based on the given passages. "
    "Only give the answer and do not output any other words.\n\n"
    "Passages:\n"
)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def build_final_prompt(doc_prompts: List[str], query_text: str) -> str:
    return PREFIX_PROMPT + "".join(doc_prompts) + query_text


def build_chunk_request(session_id: str, doc_prompts: List[str], query_text: str,
                        cfg: RuntimeConfig, use_cache: bool,
                        recompute_strategy: str,
                        recomp_ratio: float = 0.0,
                        qaw_type: str = "packed",
                        max_new_tokens: Optional[int] = None) -> GenerateRequest:
    return GenerateRequest(
        session_id=session_id,
        prompt=build_final_prompt(doc_prompts, query_text),
        max_new_tokens=cfg.max_new_tokens if max_new_tokens is None else max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        use_cache=use_cache,
        recompute_strategy=recompute_strategy,
        recomp_ratio=recomp_ratio,
        qaw_type=qaw_type,
        query_text=query_text,
        prefix_text=PREFIX_PROMPT,
        chunk_texts=doc_prompts,
        suffix_text=query_text,
        chunk_namespace="blend_demo",
    )


def warm_chunk_cache(engine: InferenceEngine, doc_prompts: List[str],
                     query_text: str, cfg: RuntimeConfig, session_id: str):
    warm_req = build_chunk_request(
        session_id=session_id,
        doc_prompts=doc_prompts,
        query_text=query_text,
        cfg=cfg,
        use_cache=True,
        recompute_strategy="none",
        max_new_tokens=0,
    )
    return engine.generate(warm_req)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chunk-cache blend demo")
    parser.add_argument("--model-name",
                        type=str,
                        default="",
                        help="显式指定模型路径/名称；未传时回退 MODEL_NAME")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--qaw-ratio", type=float, default=0.5)
    parser.add_argument("--qaw-type",
                        choices=["packed", "window"],
                        default="packed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RuntimeConfig()
    if args.model_name.strip():
        cfg.model_name = args.model_name.strip()
    cfg.max_new_tokens = 10

    output_writer = ExperimentOutputWriter.create(
        os.path.join(ROOT_DIR, "outputs"), run_tag="blend")
    console_log_path = output_writer.file_path.replace(".output",
                                                        "_console.output")
    logger = setup_logger("blend_mvp", cfg.log_level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(console_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.propagate = False

    model_runner = HFModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger, cfg.kv_max_chunks)
    engine = InferenceEngine(model_runner, kv_cache, logger)

    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend.py",
        "dataset_name": "blend_demo",
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "qaw_ratio": args.qaw_ratio,
        "qaw_type": args.qaw_type,
    })

    reuse_ttft_list: List[float] = []
    qaw_ttft_list: List[float] = []
    base_ttft_list: List[float] = []
    reuse_total_list: List[float] = []
    qaw_total_list: List[float] = []
    base_total_list: List[float] = []

    for sample_idx in range(1, args.count + 1):
        input_path = os.path.join(ROOT_DIR, "inputs", f"{sample_idx}.json")
        with open(input_path, "r", encoding="utf-8") as f:
            ex = json.load(f)
        chunk_num = ex["chunk_num"]
        doc_prompts = [ex[str(i)] for i in range(chunk_num)]
        query_text = ex["query"]

        kv_cache.clear_chunks("blend_demo")
        warm_res = warm_chunk_cache(engine, doc_prompts, query_text, cfg,
                                   f"blend-warm-chunks-{sample_idx}")

        base_res = engine.generate(
            build_chunk_request(
                session_id=f"blend-prefill-{sample_idx}",
                doc_prompts=doc_prompts,
                query_text=query_text,
                cfg=cfg,
                use_cache=False,
                recompute_strategy="none",
            ))
        reuse_res = engine.generate(
            build_chunk_request(
                session_id=f"blend-reuse-{sample_idx}",
                doc_prompts=doc_prompts,
                query_text=query_text,
                cfg=cfg,
                use_cache=True,
                recompute_strategy="none",
            ))
        qaw_res = engine.generate(
            build_chunk_request(
                session_id=f"blend-qaw-{sample_idx}",
                doc_prompts=doc_prompts,
                query_text=query_text,
                cfg=cfg,
                use_cache=True,
                recompute_strategy="query_aware",
                recomp_ratio=args.qaw_ratio,
                qaw_type=args.qaw_type,
            ))

        reuse_ttft_list.append(reuse_res.first_token_latency_s)
        qaw_ttft_list.append(qaw_res.first_token_latency_s)
        base_ttft_list.append(base_res.first_token_latency_s)
        reuse_total_list.append(reuse_res.total_latency_s)
        qaw_total_list.append(qaw_res.total_latency_s)
        base_total_list.append(base_res.total_latency_s)

        output_writer.append_json({
            "event": "sample_result",
            "sample_idx": sample_idx,
            "chunk_num": chunk_num,
            "warm_chunk_cache_hits": warm_res.chunk_cache_hits,
            "warm_chunk_cache_misses": warm_res.chunk_cache_misses,
            "full_prefill": {
                "generated_text": base_res.generated_text,
                "ttft_s": base_res.first_token_latency_s,
                "total_s": base_res.total_latency_s,
            },
            "full_reuse": {
                "generated_text": reuse_res.generated_text,
                "ttft_s": reuse_res.first_token_latency_s,
                "total_s": reuse_res.total_latency_s,
                "chunk_cache_hits": reuse_res.chunk_cache_hits,
                "chunk_cache_misses": reuse_res.chunk_cache_misses,
            },
            "query_aware": {
                "generated_text": qaw_res.generated_text,
                "ttft_s": qaw_res.first_token_latency_s,
                "total_s": qaw_res.total_latency_s,
                "recomputed_tokens": qaw_res.recomputed_tokens,
                "chunk_cache_hits": qaw_res.chunk_cache_hits,
                "chunk_cache_misses": qaw_res.chunk_cache_misses,
            },
        })
        kv_cache.clear_chunks("blend_demo")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_writer.append_json({
        "event": "run_summary",
        "sample_count": args.count,
        "full_reuse_avg_ttft_s": _mean(reuse_ttft_list),
        "query_aware_avg_ttft_s": _mean(qaw_ttft_list),
        "full_prefill_avg_ttft_s": _mean(base_ttft_list),
        "full_reuse_avg_total_s": _mean(reuse_total_list),
        "query_aware_avg_total_s": _mean(qaw_total_list),
        "full_prefill_avg_total_s": _mean(base_total_list),
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
