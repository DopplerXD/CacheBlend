"""SAMSum chunk-cache 实验：full_prefill / full_reuse / query_aware。"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from typing import List, Optional

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from cache.kv_cache import KVCacheManager
from config import RuntimeConfig
from engine.inference_engine import InferenceEngine
from model.hf_model import HFModelRunner
from schema.types import GenerateRequest
from utils.experiment_output import ExperimentOutputWriter, utc8_now_str
from utils.logging_utils import setup_logger

_EXAMPLE_UTILS_PATH = os.path.join(EXAMPLE_DIR, "utils.py")
_EXAMPLE_UTILS_SPEC = importlib.util.spec_from_file_location(
    "cacheblend_example_utils", _EXAMPLE_UTILS_PATH)
if _EXAMPLE_UTILS_SPEC is None or _EXAMPLE_UTILS_SPEC.loader is None:
    raise ImportError(f"无法加载 example/utils.py: {_EXAMPLE_UTILS_PATH}")
_EXAMPLE_UTILS_MODULE = importlib.util.module_from_spec(_EXAMPLE_UTILS_SPEC)
_EXAMPLE_UTILS_SPEC.loader.exec_module(_EXAMPLE_UTILS_MODULE)

build_fewshot_prompt = _EXAMPLE_UTILS_MODULE.build_fewshot_prompt
compute_rl = _EXAMPLE_UTILS_MODULE.compute_rl


PREFIX_PROMPT = (
    "Summarize the dialogue into a few short sentences. "
    "The following are some examples.\n\n"
)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_max_rougel(pred_text: str, answers: List[str]) -> Optional[float]:
    if not answers:
        return None
    try:
        return max([compute_rl(pred_text, answer) for answer in answers])
    except Exception:
        return None


def build_prefix_prompt(doc_prompts: List[str]) -> str:
    return PREFIX_PROMPT + "".join(doc_prompts)


def build_final_prompt(doc_prompts: List[str], q_prompt: str) -> str:
    return build_prefix_prompt(doc_prompts) + q_prompt


def build_chunk_request(session_id: str, doc_prompts: List[str], q_prompt: str,
                        query_text: str, cfg: RuntimeConfig, use_cache: bool,
                        recompute_strategy: str,
                        recomp_ratio: float = 0.0,
                        qaw_type: str = "packed",
                        max_new_tokens: Optional[int] = None) -> GenerateRequest:
    final_prompt = build_final_prompt(doc_prompts, q_prompt)
    return GenerateRequest(
        session_id=session_id,
        prompt=final_prompt,
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
        suffix_text=q_prompt,
        chunk_namespace="samsum",
    )


def warm_chunk_cache(engine: InferenceEngine, doc_prompts: List[str],
                     q_prompt: str, query_text: str, cfg: RuntimeConfig,
                     session_id: str):
    warm_req = build_chunk_request(
        session_id=session_id,
        doc_prompts=doc_prompts,
        q_prompt=q_prompt,
        query_text=query_text,
        cfg=cfg,
        use_cache=True,
        recompute_strategy="none",
        max_new_tokens=0,
    )
    return engine.generate(warm_req)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAMSum chunk-cache 对比实验脚本")
    parser.add_argument("--model-name",
                        type=str,
                        default="",
                        help="显式指定模型路径/名称；未传时回退 MODEL_NAME")
    parser.add_argument("--count", type=int, default=100, help="最多评测样本数")
    parser.add_argument("--qaw-ratio", type=float, default=0.7, help="QAW chunk token 重算比例")
    parser.add_argument("--qaw-type",
                        choices=["packed", "window"],
                        default="packed",
                        help="QAW 重算方式：packed 为原实现，window 为左侧16token窗口重算")
    parser.add_argument("--output-dir",
                        type=str,
                        default=os.path.join(ROOT_DIR, "outputs"),
                        help="实验日志输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RuntimeConfig()
    if args.model_name.strip():
        cfg.model_name = args.model_name.strip()
    cfg.max_new_tokens = 64

    logger = setup_logger("blend_samsum", cfg.log_level)
    logger.disabled = True

    model_runner = HFModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger, cfg.kv_max_chunks)
    engine = InferenceEngine(model_runner, kv_cache, logger)

    output_writer = ExperimentOutputWriter.create(args.output_dir, run_tag="samsum")
    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend_samsum.py",
        "dataset": "inputs/samsum.json",
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "count": args.count,
        "qaw_ratio": args.qaw_ratio,
        "qaw_type": args.qaw_type,
        "metric": "rougeL",
    })

    dataset_path = os.path.join(ROOT_DIR, "inputs", "samsum.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    qaw_ttft_list: List[float] = []
    base_ttft_list: List[float] = []
    reuse_ttft_list: List[float] = []
    qaw_total_list: List[float] = []
    base_total_list: List[float] = []
    reuse_total_list: List[float] = []
    qaw_rl_list: List[float] = []
    base_rl_list: List[float] = []
    reuse_rl_list: List[float] = []
    qaw_true_recompute_count = 0

    count = 0
    for sample_idx, ex in enumerate(eval_dataset, start=1):
        count += 1
        answers = list(ex.get("answers", []))
        doc_prompts, q_prompt = build_fewshot_prompt(ex)
        query_text = (ex.get("question", "") or "").strip()

        kv_cache.clear_chunks("samsum")
        warm_res = warm_chunk_cache(engine, doc_prompts, q_prompt, query_text, cfg,
                                   f"samsum-warm-chunks-{sample_idx}")

        base_req = build_chunk_request(
            session_id=f"samsum-baseline-{sample_idx}",
            doc_prompts=doc_prompts,
            q_prompt=q_prompt,
            query_text=query_text,
            cfg=cfg,
            use_cache=False,
            recompute_strategy="none",
        )
        base_res = engine.generate(base_req)

        reuse_req = build_chunk_request(
            session_id=f"samsum-reuse-{sample_idx}",
            doc_prompts=doc_prompts,
            q_prompt=q_prompt,
            query_text=query_text,
            cfg=cfg,
            use_cache=True,
            recompute_strategy="none",
        )
        reuse_res = engine.generate(reuse_req)

        qaw_req = build_chunk_request(
            session_id=f"samsum-qaw-{sample_idx}",
            doc_prompts=doc_prompts,
            q_prompt=q_prompt,
            query_text=query_text,
            cfg=cfg,
            use_cache=True,
            recompute_strategy="query_aware",
            recomp_ratio=args.qaw_ratio,
            qaw_type=args.qaw_type,
        )
        qaw_res = engine.generate(qaw_req)

        qaw_true_recompute = (
            qaw_res.recompute_mode == "chunk_query_aware_recompute"
            and qaw_res.recomputed_tokens > 0)
        if qaw_true_recompute:
            qaw_true_recompute_count += 1

        reuse_ttft_list.append(reuse_res.first_token_latency_s)
        qaw_ttft_list.append(qaw_res.first_token_latency_s)
        base_ttft_list.append(base_res.first_token_latency_s)
        reuse_total_list.append(reuse_res.total_latency_s)
        qaw_total_list.append(qaw_res.total_latency_s)
        base_total_list.append(base_res.total_latency_s)

        reuse_rl = _safe_max_rougel(reuse_res.generated_text, answers)
        qaw_rl = _safe_max_rougel(qaw_res.generated_text, answers)
        base_rl = _safe_max_rougel(base_res.generated_text, answers)
        if reuse_rl is not None:
            reuse_rl_list.append(reuse_rl)
        if qaw_rl is not None:
            qaw_rl_list.append(qaw_rl)
        if base_rl is not None:
            base_rl_list.append(base_rl)

        output_writer.append_json({
            "event": "sample_result",
            "sample_idx": sample_idx,
            "chunk_num": len(doc_prompts),
            "answers": answers,
            "warm_chunk_count": warm_res.chunk_count,
            "warm_chunk_cache_hits": warm_res.chunk_cache_hits,
            "warm_chunk_cache_misses": warm_res.chunk_cache_misses,
            "full_reuse": {
                "generated_text": reuse_res.generated_text,
                "ttft_s": reuse_res.first_token_latency_s,
                "total_s": reuse_res.total_latency_s,
                "reused_prefix_tokens": reuse_res.reused_prefix_tokens,
                "chunk_cache_hits": reuse_res.chunk_cache_hits,
                "chunk_cache_misses": reuse_res.chunk_cache_misses,
                "recompute_mode": reuse_res.recompute_mode,
                "rougeL": reuse_rl,
            },
            "query_aware": {
                "generated_text": qaw_res.generated_text,
                "ttft_s": qaw_res.first_token_latency_s,
                "total_s": qaw_res.total_latency_s,
                "reused_prefix_tokens": qaw_res.reused_prefix_tokens,
                "recomputed_tokens": qaw_res.recomputed_tokens,
                "chunk_cache_hits": qaw_res.chunk_cache_hits,
                "chunk_cache_misses": qaw_res.chunk_cache_misses,
                "recompute_mode": qaw_res.recompute_mode,
                "true_recompute": qaw_true_recompute,
                "rougeL": qaw_rl,
            },
            "full_prefill": {
                "generated_text": base_res.generated_text,
                "ttft_s": base_res.first_token_latency_s,
                "total_s": base_res.total_latency_s,
                "rougeL": base_rl,
            },
        })
        kv_cache.clear_chunks("samsum")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if count >= args.count:
            break

    output_writer.append_json({
        "event": "run_summary",
        "sample_count": count,
        "full_reuse_avg_ttft_s": _mean(reuse_ttft_list),
        "query_aware_avg_ttft_s": _mean(qaw_ttft_list),
        "full_prefill_avg_ttft_s": _mean(base_ttft_list),
        "full_reuse_avg_total_s": _mean(reuse_total_list),
        "query_aware_avg_total_s": _mean(qaw_total_list),
        "full_prefill_avg_total_s": _mean(base_total_list),
        "full_reuse_avg_rougeL": _mean(reuse_rl_list),
        "query_aware_avg_rougeL": _mean(qaw_rl_list),
        "full_prefill_avg_rougeL": _mean(base_rl_list),
        "qaw_true_recompute_count": qaw_true_recompute_count,
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
