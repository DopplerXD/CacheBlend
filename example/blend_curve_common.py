"""Curve 实验公共工具：支持 MusiQue / WikiMQA 的 QAW 网格评测。"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

# 允许脚本以 `python example/*.py` 直接运行。
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

_EXAMPLE_UTILS_PATH = os.path.join(EXAMPLE_DIR, "utils.py")
_EXAMPLE_UTILS_SPEC = importlib.util.spec_from_file_location(
    "cacheblend_example_utils", _EXAMPLE_UTILS_PATH)
if _EXAMPLE_UTILS_SPEC is None or _EXAMPLE_UTILS_SPEC.loader is None:
    raise ImportError(f"无法加载 example/utils.py: {_EXAMPLE_UTILS_PATH}")
_EXAMPLE_UTILS_MODULE = importlib.util.module_from_spec(_EXAMPLE_UTILS_SPEC)
_EXAMPLE_UTILS_SPEC.loader.exec_module(_EXAMPLE_UTILS_MODULE)

build_qa_prompt = _EXAMPLE_UTILS_MODULE.build_qa_prompt
build_fewshot_prompt = _EXAMPLE_UTILS_MODULE.build_fewshot_prompt
compute_f1 = _EXAMPLE_UTILS_MODULE.compute_f1
compute_rl = _EXAMPLE_UTILS_MODULE.compute_rl
normalize_question = _EXAMPLE_UTILS_MODULE.normalize_question


def _default_answers(raw_answers) -> List[str]:
    return list(raw_answers or [])


def _normalize_wikimqa_answers(raw_answers) -> List[str]:
    normalized = []
    for item in raw_answers or []:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, list):
            if item and isinstance(item[0], str):
                normalized.append(item[0])
        elif item is not None:
            normalized.append(str(item))
    return normalized


DATASET_SPECS: Dict[str, Dict] = {
    "musique": {
        "dataset_path": "inputs/musique_s.json",
        "prefix_prompt": (
            "You will be asked a question after reading several passages. "
            "Please directly answer the question based on the given passages. "
            "Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n"
        ),
        "query_prompt": (
            "\n\nAnswer the question directly based on the given passages. "
            "Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
        ),
        "answer_fn": _default_answers,
        "ratio_curve_tag": "musique_curve",
        "suffix_curve_tag": "musique_suffix_curve",
        "grid_curve_tag": "musique_ratio_suffix_curve",
    },
    "wikimqa": {
        "dataset_path": "inputs/wikimqa_s.json",
        "prefix_prompt": (
            "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words.\n\n"
            "The following are given passages.\n"
        ),
        "query_prompt": (
            "\n\nAnswer the question based on the given passages. "
            "Answer the question within 5 words. Do NOT repeat the question or output any other words. "
            "Question: "
        ),
        "answer_fn": _normalize_wikimqa_answers,
        "ratio_curve_tag": "wikimqa_curve",
        "suffix_curve_tag": "wikimqa_suffix_curve",
        "grid_curve_tag": "wikimqa_ratio_suffix_curve",
    },
    "cmrc": {
        "dataset_path": "inputs/cmrc_s.json",
        "prefix_prompt": (
            "请基于给定文章直接回答问题，只输出简短答案短语，不要输出额外内容。\n\n"
            "文章：\n"
        ),
        "query_prompt": (
            "\n\n请基于给定文章直接回答问题，只输出简短答案短语，不要重复问题。"
            "问题："
        ),
        "answer_fn": _default_answers,
        "ratio_curve_tag": "cmrc_curve",
        "suffix_curve_tag": "cmrc_suffix_curve",
        "grid_curve_tag": "cmrc_ratio_suffix_curve",
        "prefill_only_baseline": True,
    },
    "samsum": {
        "dataset_path": "inputs/samsum.json",
        "prefix_prompt": (
            "Summarize the dialogue into a few short sentences. "
            "The following are some examples.\n\n"
        ),
        "query_prompt": "",
        "answer_fn": _default_answers,
        "ratio_curve_tag": "samsum_curve",
        "suffix_curve_tag": "samsum_suffix_curve",
        "grid_curve_tag": "samsum_ratio_suffix_curve",
        "prompt_kind": "fewshot",
        "metric": "rougeL",
    },
}


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def safe_max_f1(pred_text: str, answers: List[str], tokenizer) -> Optional[float]:
    if not answers:
        return None
    try:
        return max([compute_f1(pred_text, answer, tokenizer) for answer in answers])
    except Exception:
        return 0.0


def safe_max_rougel(pred_text: str, answers: List[str]) -> Optional[float]:
    if not answers:
        return None
    try:
        return max([compute_rl(pred_text, answer) for answer in answers])
    except Exception:
        return None


def safe_max_metric(pred_text: str, answers: List[str], tokenizer,
                    metric: str) -> Optional[float]:
    if metric == "rougeL":
        return safe_max_rougel(pred_text, answers)
    return safe_max_f1(pred_text, answers, tokenizer)


def float_grid(min_value: float, max_value: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step 必须 > 0")
    lo = min(min_value, max_value)
    hi = max(min_value, max_value)
    values: List[float] = []
    current = lo
    while current <= hi + 1e-9:
        values.append(round(current, 2))
        current += step
    if not values:
        values = [round(lo, 2)]
    return values


def int_grid(min_value: int, max_value: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step 必须 > 0")
    lo = min(min_value, max_value)
    hi = max(min_value, max_value)
    return list(range(lo, hi + 1, step))


def build_prefix_prompt(prefix_prompt: str, doc_prompts: List[str]) -> str:
    return prefix_prompt + "".join(doc_prompts)


def build_final_prompt(prefix_prompt: str, doc_prompts: List[str], q_prompt: str) -> str:
    return build_prefix_prompt(prefix_prompt, doc_prompts) + q_prompt


def build_sample_parts(ex: Dict, spec: Dict) -> Tuple[List[str], str, str]:
    if spec.get("prompt_kind") == "fewshot":
        doc_prompts, q_prompt = build_fewshot_prompt(ex)
        query_text = (ex.get("question", "") or "").strip()
        return doc_prompts, q_prompt, query_text
    doc_prompts, q_prompt = build_qa_prompt(ex, spec["query_prompt"])
    query_text = normalize_question(ex.get("question", ""))
    return doc_prompts, q_prompt, query_text


def build_chunk_request(session_id: str, spec: Dict, doc_prompts: List[str],
                        q_prompt: str, query_text: str, cfg: RuntimeConfig,
                        use_cache: bool, recompute_strategy: str,
                        recomp_ratio: float = 0.0,
                        qaw_type: str = "packed",
                        qaw_variant: str = "default",
                        qaw_random_seed: int = 0,
                        max_new_tokens: Optional[int] = None) -> GenerateRequest:
    final_prompt = build_final_prompt(spec["prefix_prompt"], doc_prompts, q_prompt)
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
        qaw_variant=qaw_variant,
        qaw_random_seed=qaw_random_seed,
        query_text=query_text,
        prefix_text=spec["prefix_prompt"],
        chunk_texts=doc_prompts,
        suffix_text=q_prompt,
        chunk_namespace=str(spec.get("ratio_curve_tag", "chunk_curve")),
    )


def load_dataset(dataset_name: str) -> Tuple[Dict, List[Dict]]:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    spec = DATASET_SPECS[dataset_name]
    dataset_path = os.path.join(ROOT_DIR, spec["dataset_path"])
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)
    return spec, eval_dataset


def seed_session_cache(kv_cache: KVCacheManager, src_session_id: str,
                       dst_session_id: str) -> bool:
    src = kv_cache.get(src_session_id)
    if src is None:
        return False
    kv_cache.clear(dst_session_id)
    kv_cache.put(
        dst_session_id,
        list(src.token_ids),
        src.past_key_values,
        next_token_logits=src.next_token_logits,
    )
    return True


def warm_prompt_cache(engine: InferenceEngine, kv_cache: KVCacheManager,
                      session_id: str, prompt: str):
    kv_cache.clear(session_id)
    warm_req = GenerateRequest(
        session_id=session_id,
        prompt=prompt,
        max_new_tokens=0,
        temperature=0.0,
        top_p=1.0,
        use_cache=True,
        recompute_strategy="none",
    )
    return engine.generate(warm_req)


def warm_chunk_cache(engine: InferenceEngine, spec: Dict,
                     doc_prompts: List[str], q_prompt: str,
                     query_text: str, cfg: RuntimeConfig, session_id: str):
    warm_req = build_chunk_request(
        session_id=session_id,
        spec=spec,
        doc_prompts=doc_prompts,
        q_prompt=q_prompt,
        query_text=query_text,
        cfg=cfg,
        use_cache=True,
        recompute_strategy="none",
        max_new_tokens=0,
    )
    return engine.generate(warm_req)


def cleanup_runtime(kv_cache: Optional[KVCacheManager],
                    engine: Optional[InferenceEngine]) -> None:
    if kv_cache is not None:
        kv_cache._store.clear()  # pylint: disable=protected-access
        kv_cache._chunk_store.clear()  # pylint: disable=protected-access
    del engine
    del kv_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_cuda_mem_mb() -> Optional[Dict[str, float]]:
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated_bytes = torch.cuda.memory_allocated()
    reserved_bytes = torch.cuda.memory_reserved()
    return {
        "allocated_mb": allocated_bytes / 1024 / 1024,
        "reserved_mb": reserved_bytes / 1024 / 1024,
        "free_mb": free_bytes / 1024 / 1024,
        "total_mb": total_bytes / 1024 / 1024,
    }


def format_cuda_mem(mem: Optional[Dict[str, float]]) -> str:
    if mem is None:
        return "cuda=unavailable"
    return (
        f"allocated={mem['allocated_mb']:.1f}MB "
        f"reserved={mem['reserved_mb']:.1f}MB "
        f"free={mem['free_mb']:.1f}MB "
        f"total={mem['total_mb']:.1f}MB"
    )


def run_fixed_baselines(
    eval_dataset: List[Dict],
    spec: Dict,
    cfg: RuntimeConfig,
    model_runner: HFModelRunner,
    logger,
    sample_limit: int,
) -> Dict[str, Optional[float]]:
    """仅运行一次 full_reuse / full_prefill，供全部曲线组复用。"""
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger, cfg.kv_max_chunks)
    engine = InferenceEngine(model_runner, kv_cache, logger)

    base_ttft_list: List[float] = []
    base_total_list: List[float] = []
    base_f1_list: List[float] = []

    answer_fn = spec["answer_fn"]
    reuse_ttft_list: List[float] = []
    reuse_total_list: List[float] = []
    reuse_f1_list: List[float] = []
    metric_name = spec.get("metric", "f1")
    count = 0
    try:
        for sample_idx, ex in enumerate(eval_dataset, start=1):
            if count >= sample_limit:
                break
            count += 1

            answers = answer_fn(ex.get("answers", []))
            doc_prompts, q_prompt, query_text = build_sample_parts(ex, spec)

            # full_prefill 不依赖 chunk cache，先跑它，避免已预热的 chunk KV
            # 占用显存并抬高长 prompt prefill 的峰值。
            kv_cache.clear_chunks()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            base_req = build_chunk_request(
                session_id=f"baseline-prefill-{sample_idx}",
                spec=spec,
                doc_prompts=doc_prompts,
                q_prompt=q_prompt,
                query_text=query_text,
                cfg=cfg,
                use_cache=False,
                recompute_strategy="none",
            )
            base_res = engine.generate(base_req)

            base_ttft_list.append(base_res.first_token_latency_s)
            base_total_list.append(base_res.total_latency_s)

            base_f1 = safe_max_metric(base_res.generated_text, answers,
                                      model_runner.tokenizer, metric_name)
            if base_f1 is not None:
                base_f1_list.append(base_f1)

            del base_res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            warm_chunk_cache(engine, spec, doc_prompts, q_prompt, query_text, cfg,
                             f"baseline-warm-chunks-{sample_idx}")
            reuse_req = build_chunk_request(
                session_id=f"baseline-reuse-{sample_idx}",
                spec=spec,
                doc_prompts=doc_prompts,
                q_prompt=q_prompt,
                query_text=query_text,
                cfg=cfg,
                use_cache=True,
                recompute_strategy="none",
            )
            reuse_res = engine.generate(reuse_req)
            reuse_ttft_list.append(reuse_res.first_token_latency_s)
            reuse_total_list.append(reuse_res.total_latency_s)

            reuse_f1 = safe_max_metric(reuse_res.generated_text, answers,
                                       model_runner.tokenizer, metric_name)
            if reuse_f1 is not None:
                reuse_f1_list.append(reuse_f1)

            del reuse_res
            kv_cache.clear_chunks()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        cleanup_runtime(kv_cache, engine)

    return {
        "sample_count": count,
        "full_reuse_avg_ttft_s": mean(reuse_ttft_list),
        "full_prefill_avg_ttft_s": mean(base_ttft_list),
        "full_reuse_avg_total_s": mean(reuse_total_list),
        "full_prefill_avg_total_s": mean(base_total_list),
        "full_reuse_avg_f1": mean(reuse_f1_list),
        "full_prefill_avg_f1": mean(base_f1_list),
    }


def evaluate_qaw_grid(
    eval_dataset: List[Dict],
    spec: Dict,
    cfg: RuntimeConfig,
    model_runner: HFModelRunner,
    logger,
    sample_limit: int,
    ratio_values: List[float],
    suffix_values: List[int],
    qaw_type: str = "packed",
    qaw_variant: str = "default",
    qaw_random_seed: int = 0,
) -> Tuple[int, Dict[Tuple[float, int], Dict[str, object]]]:
    grid_pairs = [(ratio, suffix) for ratio in ratio_values for suffix in suffix_values]
    total_groups = len(grid_pairs)
    metrics: Dict[Tuple[float, int], Dict[str, object]] = {
        pair: {
            "ttft": [],
            "total": [],
            "f1": [],
            "true_recompute": 0,
        }
        for pair in grid_pairs
    }

    for group_idx, (recomp_ratio, suffix_len) in enumerate(grid_pairs, start=1):
        print(
            "[curve] 进行中 "
            f"{group_idx}/{total_groups}, recomp_ratio={recomp_ratio:.2f}, "
            f"suffix_len={suffix_len}",
            flush=True,
        )

    processed_samples = 0
    answer_fn = spec["answer_fn"]
    metric_name = spec.get("metric", "f1")
    for sample_idx, ex in enumerate(eval_dataset, start=1):
        if processed_samples >= sample_limit:
            break
        processed_samples += 1
        print(f"[curve] 样本 {processed_samples}/{sample_limit} 开始", flush=True)

        answers = answer_fn(ex.get("answers", []))
        doc_prompts, q_prompt, query_text = build_sample_parts(ex, spec)

        kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger, cfg.kv_max_chunks)
        engine = InferenceEngine(model_runner, kv_cache, logger)

        try:
            warm_chunk_cache(engine, spec, doc_prompts, q_prompt, query_text, cfg,
                             f"sample-{sample_idx}-warm-chunks")
            for recomp_ratio, suffix_len in grid_pairs:
                ratio_tag = f"{recomp_ratio:.2f}".replace(".", "p")
                qaw_req = build_chunk_request(
                    session_id=f"sample-{sample_idx}-qaw-r{ratio_tag}",
                    spec=spec,
                    doc_prompts=doc_prompts,
                    q_prompt=q_prompt,
                    query_text=query_text,
                    cfg=cfg,
                    use_cache=True,
                    recompute_strategy="query_aware",
                    recomp_ratio=recomp_ratio,
                    qaw_type=qaw_type,
                    qaw_variant=qaw_variant,
                    qaw_random_seed=qaw_random_seed,
                )
                qaw_res = engine.generate(qaw_req)

                bucket = metrics[(recomp_ratio, suffix_len)]
                bucket["ttft"].append(qaw_res.first_token_latency_s)
                bucket["total"].append(qaw_res.total_latency_s)
                if (qaw_res.recompute_mode == "chunk_query_aware_recompute"
                        and qaw_res.recomputed_tokens > 0):
                    bucket["true_recompute"] += 1

                qaw_f1 = safe_max_metric(qaw_res.generated_text, answers,
                                         model_runner.tokenizer, metric_name)
                if qaw_f1 is not None:
                    bucket["f1"].append(qaw_f1)
        finally:
            cleanup_runtime(kv_cache, engine)
            mem_after_cleanup = get_cuda_mem_mb()
            print(
                f"[curve] 样本 {processed_samples}/{sample_limit} 完成, "
                f"已清理显存: {format_cuda_mem(mem_after_cleanup)}",
                flush=True,
            )

    return processed_samples, metrics


def build_run_summary_payload(processed_samples: int,
                              baseline_stats: Dict[str, Optional[float]],
                              grid_bucket: Dict[str, object], recomp_ratio: float,
                              suffix_len: int) -> Dict[str, object]:
    return {
        "event": "run_summary",
        "recomp_ratio": recomp_ratio,
        "suffix_len": suffix_len,
        "sample_count": processed_samples,
        "full_reuse_avg_ttft_s": baseline_stats["full_reuse_avg_ttft_s"],
        "query_aware_avg_ttft_s": mean(grid_bucket["ttft"]),
        "full_prefill_avg_ttft_s": baseline_stats["full_prefill_avg_ttft_s"],
        "full_reuse_avg_total_s": baseline_stats["full_reuse_avg_total_s"],
        "query_aware_avg_total_s": mean(grid_bucket["total"]),
        "full_prefill_avg_total_s": baseline_stats["full_prefill_avg_total_s"],
        "full_reuse_avg_f1": baseline_stats["full_reuse_avg_f1"],
        "query_aware_avg_f1": mean(grid_bucket["f1"]),
        "full_prefill_avg_f1": baseline_stats["full_prefill_avg_f1"],
        "qaw_true_recompute_count": grid_bucket["true_recompute"],
    }
