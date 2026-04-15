"""统一实验 Runner：支持 full_reuse/full_prefill 与 qaw 子变体。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Callable, Dict, List, Optional

# 允许从项目根目录导入模块（保持 `python example/blend_runner.py` 可直接运行）。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from cache.kv_cache import KVCacheManager
from config import RuntimeConfig
from engine.inference_engine import InferenceEngine
from model.yi_model import YiModelRunner
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

build_qa_prompt = _EXAMPLE_UTILS_MODULE.build_qa_prompt
compute_f1 = _EXAMPLE_UTILS_MODULE.compute_f1
normalize_question = _EXAMPLE_UTILS_MODULE.normalize_question


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_max_f1(pred_text: str, answers: List[str], tokenizer) -> Optional[float]:
    """稳健 F1：空输出或异常时返回 0，避免单样本中断整次实验。"""
    if not answers:
        return None
    try:
        return max([compute_f1(pred_text, a, tokenizer) for a in answers])
    except Exception:
        return 0.0


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


def _default_answers(raw_answers) -> List[str]:
    return list(raw_answers or [])


DATASET_SPECS: Dict[str, Dict] = {
    "musique": {
        "dataset_path": "inputs/musique_s.json",
        "run_tag": "runner_musique",
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
    },
    "wikimqa": {
        "dataset_path": "inputs/wikimqa_s.json",
        "run_tag": "runner_wikimqa",
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
    },
    "squad": {
        "dataset_path": "inputs/squad_s.json",
        "run_tag": "runner_squad",
        "prefix_prompt": (
            "Answer the question based on the given passage. "
            "Only output the short answer phrase.\n\n"
            "Passage:\n"
        ),
        "query_prompt": (
            "\n\nAnswer the question based on the given passage. "
            "Only output the short answer phrase. "
            "Question: "
        ),
        "answer_fn": _default_answers,
    },
}


def build_prefix_prompt(prefix_prompt: str, doc_prompts: List[str]) -> str:
    return prefix_prompt + "".join(doc_prompts)


def build_final_prompt(prefix_prompt: str, doc_prompts: List[str], q_prompt: str) -> str:
    return build_prefix_prompt(prefix_prompt, doc_prompts) + q_prompt


def build_stale_prompt(prefix_prompt: str, query_prompt: str, doc_prompts: List[str],
                       query_text: str) -> str:
    stale_q_prompt = (
        f"{query_prompt} Placeholder cache-warm question: {query_text}\nAnswer:"
    )
    return build_final_prompt(prefix_prompt, doc_prompts, stale_q_prompt)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一实验 runner（含 qaw 子变体）")
    parser.add_argument("--dataset",
                        choices=sorted(DATASET_SPECS.keys()),
                        default="musique")
    parser.add_argument("--count", type=int, default=30, help="最多评测样本数")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--qaw-ratio", type=float, default=0.30)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--qaw-random-seed", type=int, default=2026)
    parser.add_argument("--summary-only",
                        action="store_true",
                        help="只写 run_summary，不写 sample_result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    answer_fn: Callable = spec["answer_fn"]

    cfg = RuntimeConfig()
    cfg.max_new_tokens = args.max_new_tokens

    logger = setup_logger(f"blend_runner_{args.dataset}", cfg.log_level)
    logger.disabled = True

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype, logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
    engine = InferenceEngine(model_runner, kv_cache, logger)

    output_writer = ExperimentOutputWriter.create(
        os.path.join(ROOT_DIR, "outputs"), run_tag=spec["run_tag"])
    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend_runner.py",
        "dataset": spec["dataset_path"],
        "dataset_name": args.dataset,
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "qaw_ratio": args.qaw_ratio,
        "suffix_len": args.suffix_len,
        "qaw_random_seed": args.qaw_random_seed,
    })

    dataset_path = os.path.join(ROOT_DIR, spec["dataset_path"])
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    metrics = {
        "full_prefill": {
            "ttft": [],
            "total": [],
            "f1": [],
            "true_recompute": 0
        },
        "full_reuse": {
            "ttft": [],
            "total": [],
            "f1": [],
            "true_recompute": 0
        },
        "qaw_default": {
            "ttft": [],
            "total": [],
            "f1": [],
            "true_recompute": 0
        },
        "qaw_no_suffix": {
            "ttft": [],
            "total": [],
            "f1": [],
            "true_recompute": 0
        },
        "qaw_random_topk": {
            "ttft": [],
            "total": [],
            "f1": [],
            "true_recompute": 0
        },
    }

    qaw_variant_cfgs = [
        ("qaw_default", "default", args.suffix_len, 0),
        ("qaw_no_suffix", "no_suffix", 0, 0),
        ("qaw_random_topk", "random_topk", args.suffix_len, args.qaw_random_seed),
    ]

    count = 0
    for sample_idx, ex in enumerate(eval_dataset, start=1):
        if count >= args.count:
            break
        count += 1

        answers = answer_fn(ex.get("answers", []))
        doc_prompts, q_prompt = build_qa_prompt(ex, spec["query_prompt"])
        query_text = normalize_question(ex.get("question", ""))
        final_prompt = build_final_prompt(spec["prefix_prompt"], doc_prompts, q_prompt)
        stale_prompt = build_stale_prompt(spec["prefix_prompt"], spec["query_prompt"],
                                          doc_prompts, query_text)

        # 1) full prefill
        base_req = GenerateRequest(
            session_id=f"{args.dataset}-baseline-{sample_idx}",
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=False,
            recompute_strategy="none",
        )
        base_res = engine.generate(base_req)

        # 2) full reuse
        reuse_template_id = f"{args.dataset}-reuse-template-{sample_idx}"
        warm_prompt_cache(engine, kv_cache, reuse_template_id, final_prompt)
        reuse_session_id = f"{args.dataset}-reuse-{sample_idx}"
        seed_session_cache(kv_cache, reuse_template_id, reuse_session_id)
        reuse_req = GenerateRequest(
            session_id=reuse_session_id,
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=True,
            recompute_strategy="none",
        )
        reuse_res = engine.generate(reuse_req)

        # 3) stale cache template for qaw variants
        stale_template_id = f"{args.dataset}-stale-template-{sample_idx}"
        stale_warm_res = warm_prompt_cache(engine, kv_cache, stale_template_id, stale_prompt)

        # 4) qaw variants
        qaw_results: Dict[str, object] = {}
        for metric_key, qaw_variant, suffix_len, random_seed in qaw_variant_cfgs:
            sid = f"{args.dataset}-{metric_key}-{sample_idx}"
            seed_session_cache(kv_cache, stale_template_id, sid)
            req = GenerateRequest(
                session_id=sid,
                prompt=final_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                use_cache=True,
                recompute_strategy="query_aware",
                recomp_ratio=args.qaw_ratio,
                suffix_len=suffix_len,
                query_text=query_text,
                qaw_variant=qaw_variant,
                qaw_random_seed=random_seed,
            )
            qaw_results[metric_key] = engine.generate(req)

        base_f1 = _safe_max_f1(base_res.generated_text, answers, model_runner.tokenizer)
        reuse_f1 = _safe_max_f1(reuse_res.generated_text, answers, model_runner.tokenizer)

        metrics["full_prefill"]["ttft"].append(base_res.first_token_latency_s)
        metrics["full_prefill"]["total"].append(base_res.total_latency_s)
        if base_f1 is not None:
            metrics["full_prefill"]["f1"].append(base_f1)

        metrics["full_reuse"]["ttft"].append(reuse_res.first_token_latency_s)
        metrics["full_reuse"]["total"].append(reuse_res.total_latency_s)
        if reuse_f1 is not None:
            metrics["full_reuse"]["f1"].append(reuse_f1)

        sample_qaw_payload = {}
        for metric_key, _, _, _ in qaw_variant_cfgs:
            qaw_res = qaw_results[metric_key]
            qaw_true_recompute = (qaw_res.recompute_mode == "query_aware_recompute"
                                  and qaw_res.recomputed_tokens > 0)
            if qaw_true_recompute:
                metrics[metric_key]["true_recompute"] += 1

            qaw_f1 = _safe_max_f1(qaw_res.generated_text, answers, model_runner.tokenizer)
            metrics[metric_key]["ttft"].append(qaw_res.first_token_latency_s)
            metrics[metric_key]["total"].append(qaw_res.total_latency_s)
            if qaw_f1 is not None:
                metrics[metric_key]["f1"].append(qaw_f1)
            sample_qaw_payload[metric_key] = {
                "generated_text": qaw_res.generated_text,
                "ttft_s": qaw_res.first_token_latency_s,
                "total_s": qaw_res.total_latency_s,
                "recomputed_tokens": qaw_res.recomputed_tokens,
                "recompute_mode": qaw_res.recompute_mode,
                "true_recompute": qaw_true_recompute,
                "f1": qaw_f1,
            }

        if not args.summary_only:
            output_writer.append_json({
                "event": "sample_result",
                "sample_idx": sample_idx,
                "question": ex.get("question", ""),
                "answers": answers,
                "stale_cache_prompt_tokens": stale_warm_res.prompt_tokens,
                "full_prefill": {
                    "generated_text": base_res.generated_text,
                    "ttft_s": base_res.first_token_latency_s,
                    "total_s": base_res.total_latency_s,
                    "f1": base_f1,
                },
                "full_reuse": {
                    "generated_text": reuse_res.generated_text,
                    "ttft_s": reuse_res.first_token_latency_s,
                    "total_s": reuse_res.total_latency_s,
                    "reused_prefix_tokens": reuse_res.reused_prefix_tokens,
                    "f1": reuse_f1,
                },
                "qaw_variants": sample_qaw_payload,
            })

    output_writer.append_json({
        "event": "run_summary",
        "sample_count": count,
        "full_prefill_avg_ttft_s": _mean(metrics["full_prefill"]["ttft"]),
        "full_reuse_avg_ttft_s": _mean(metrics["full_reuse"]["ttft"]),
        "qaw_default_avg_ttft_s": _mean(metrics["qaw_default"]["ttft"]),
        "qaw_no_suffix_avg_ttft_s": _mean(metrics["qaw_no_suffix"]["ttft"]),
        "qaw_random_topk_avg_ttft_s": _mean(metrics["qaw_random_topk"]["ttft"]),
        "full_prefill_avg_total_s": _mean(metrics["full_prefill"]["total"]),
        "full_reuse_avg_total_s": _mean(metrics["full_reuse"]["total"]),
        "qaw_default_avg_total_s": _mean(metrics["qaw_default"]["total"]),
        "qaw_no_suffix_avg_total_s": _mean(metrics["qaw_no_suffix"]["total"]),
        "qaw_random_topk_avg_total_s": _mean(metrics["qaw_random_topk"]["total"]),
        "full_prefill_avg_f1": _mean(metrics["full_prefill"]["f1"]),
        "full_reuse_avg_f1": _mean(metrics["full_reuse"]["f1"]),
        "qaw_default_avg_f1": _mean(metrics["qaw_default"]["f1"]),
        "qaw_no_suffix_avg_f1": _mean(metrics["qaw_no_suffix"]["f1"]),
        "qaw_random_topk_avg_f1": _mean(metrics["qaw_random_topk"]["f1"]),
        "qaw_default_true_recompute_count": metrics["qaw_default"]["true_recompute"],
        "qaw_no_suffix_true_recompute_count": metrics["qaw_no_suffix"]["true_recompute"],
        "qaw_random_topk_true_recompute_count":
        metrics["qaw_random_topk"]["true_recompute"],
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
