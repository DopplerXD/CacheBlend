"""WikiMQA 速度-质量曲线脚本：仅输出每个 qaw_ratio 的 run_summary。"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional

import torch

# 允许从项目根目录导入模块（保持 `python example/blend_wikimqa_curve.py` 可直接运行）。
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


PREFIX_PROMPT = (
    "Answer the question based on the given passages. "
    "Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n"
)
QUERY_PROMPT = (
    "\n\nAnswer the question based on the given passages. "
    "Answer the question within 5 words. Do NOT repeat the question or output any other words. "
    "Question: "
)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_max_f1(pred_text: str, answers: List[str], tokenizer) -> Optional[float]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WikiMQA curve（禁用 KVD）")
    parser.add_argument("--count", type=int, default=30, help="最多评测样本数")
    parser.add_argument("--qaw-ratio-min", type=float, default=0.5)
    parser.add_argument("--qaw-ratio-max", type=float, default=0.5)
    parser.add_argument("--qaw-ratio-step", type=float, default=0.05)
    parser.add_argument("--suffix-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir",
                        type=str,
                        default=os.path.join(ROOT_DIR, "outputs"),
                        help="实验日志输出目录")
    parser.add_argument("--run-tag",
                        type=str,
                        default="wikimqa_curve",
                        help="输出文件 run_tag")
    return parser.parse_args()


def ratio_grid(min_ratio: float, max_ratio: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("qaw-ratio-step 必须 > 0")
    lo = min(min_ratio, max_ratio)
    hi = max(min_ratio, max_ratio)
    ratios: List[float] = []
    cur = lo
    while cur <= hi + 1e-9:
        ratios.append(round(cur, 2))
        cur += step
    if not ratios:
        ratios = [round(lo, 2)]
    return ratios


def build_prefix_prompt(doc_prompts: List[str]) -> str:
    return PREFIX_PROMPT + "".join(doc_prompts)


def build_final_prompt(doc_prompts: List[str], q_prompt: str) -> str:
    return build_prefix_prompt(doc_prompts) + q_prompt


def build_stale_prompt(doc_prompts: List[str], query_text: str) -> str:
    stale_q_prompt = (
        f"{QUERY_PROMPT} Placeholder cache-warm question: {query_text}\nAnswer:"
    )
    return build_final_prompt(doc_prompts, stale_q_prompt)


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


def cleanup_runtime(kv_cache: Optional[KVCacheManager],
                    engine: Optional[InferenceEngine]) -> None:
    if kv_cache is not None:
        kv_cache._store.clear()  # pylint: disable=protected-access
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
    cfg: RuntimeConfig,
    model_runner: YiModelRunner,
    logger,
    sample_limit: int,
) -> Dict[str, Optional[float]]:
    """仅运行一次 full_reuse / full_prefill，供全部 ratio 复用。"""
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
    engine = InferenceEngine(model_runner, kv_cache, logger)

    reuse_ttft_list: List[float] = []
    base_ttft_list: List[float] = []
    reuse_total_list: List[float] = []
    base_total_list: List[float] = []
    reuse_f1_list: List[float] = []
    base_f1_list: List[float] = []

    count = 0
    try:
        for sample_idx, ex in enumerate(eval_dataset, start=1):
            if count >= sample_limit:
                break
            count += 1

            answers = _normalize_wikimqa_answers(ex.get("answers", []))
            doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
            final_prompt = build_final_prompt(doc_prompts, q_prompt)

            # 1) full prefill
            base_req = GenerateRequest(
                session_id=f"baseline-prefill-{sample_idx}",
                prompt=final_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                use_cache=False,
                recompute_strategy="none",
            )
            base_res = engine.generate(base_req)

            # 2) full reuse
            reuse_template_id = f"baseline-reuse-template-{sample_idx}"
            warm_prompt_cache(engine, kv_cache, reuse_template_id, final_prompt)
            reuse_session_id = f"baseline-reuse-{sample_idx}"
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

            reuse_ttft_list.append(reuse_res.first_token_latency_s)
            base_ttft_list.append(base_res.first_token_latency_s)
            reuse_total_list.append(reuse_res.total_latency_s)
            base_total_list.append(base_res.total_latency_s)

            reuse_f1 = _safe_max_f1(reuse_res.generated_text, answers,
                                    model_runner.tokenizer)
            base_f1 = _safe_max_f1(base_res.generated_text, answers,
                                   model_runner.tokenizer)
            if reuse_f1 is not None:
                reuse_f1_list.append(reuse_f1)
            if base_f1 is not None:
                base_f1_list.append(base_f1)
    finally:
        cleanup_runtime(kv_cache, engine)

    return {
        "sample_count": count,
        "full_reuse_avg_ttft_s": _mean(reuse_ttft_list),
        "kv_diff_avg_ttft_s": None,
        "full_prefill_avg_ttft_s": _mean(base_ttft_list),
        "full_reuse_avg_total_s": _mean(reuse_total_list),
        "kv_diff_avg_total_s": None,
        "full_prefill_avg_total_s": _mean(base_total_list),
        "full_reuse_avg_f1": _mean(reuse_f1_list),
        "kv_diff_avg_f1": None,
        "full_prefill_avg_f1": _mean(base_f1_list),
        "kvd_true_recompute_count": 0,
    }


def main() -> None:
    args = parse_args()

    cfg = RuntimeConfig()
    cfg.max_new_tokens = args.max_new_tokens

    logger = setup_logger("blend_wikimqa_curve", cfg.log_level)
    logger.disabled = True

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    output_writer = ExperimentOutputWriter.create(args.output_dir,
                                                  run_tag=args.run_tag)

    ratios = ratio_grid(args.qaw_ratio_min, args.qaw_ratio_max,
                        args.qaw_ratio_step)
    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend_wikimqa_curve.py",
        "dataset": "inputs/wikimqa_s.json",
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "count": args.count,
        "qaw_ratio_min": args.qaw_ratio_min,
        "qaw_ratio_max": args.qaw_ratio_max,
        "qaw_ratio_step": args.qaw_ratio_step,
        "suffix_len": args.suffix_len,
        "ratio_group_count": len(ratios),
        "kvd_enabled": False,
    })

    dataset_path = os.path.join(ROOT_DIR, "inputs", "wikimqa_s.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    sample_limit = min(len(eval_dataset), args.count)
    baseline_stats = run_fixed_baselines(
        eval_dataset=eval_dataset,
        cfg=cfg,
        model_runner=model_runner,
        logger=logger,
        sample_limit=sample_limit,
    )

    total_groups = len(ratios)
    qaw_ttft_by_ratio: Dict[float, List[float]] = {r: [] for r in ratios}
    qaw_total_by_ratio: Dict[float, List[float]] = {r: [] for r in ratios}
    qaw_f1_by_ratio: Dict[float, List[float]] = {r: [] for r in ratios}
    qaw_true_recompute_by_ratio: Dict[float, int] = {r: 0 for r in ratios}

    for group_idx, recomp_ratio in enumerate(ratios, start=1):
        print(
            f"[curve] 进行中 {group_idx}/{total_groups}, recomp_ratio={recomp_ratio:.2f}",
            flush=True,
        )

    processed_samples = 0
    for sample_idx, ex in enumerate(eval_dataset, start=1):
        if processed_samples >= sample_limit:
            break
        processed_samples += 1
        print(f"[curve] 样本 {processed_samples}/{sample_limit} 开始", flush=True)

        answers = _normalize_wikimqa_answers(ex.get("answers", []))
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        query_text = normalize_question(ex.get("question", ""))
        final_prompt = build_final_prompt(doc_prompts, q_prompt)
        stale_prompt = build_stale_prompt(doc_prompts, query_text)

        kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
        engine = InferenceEngine(model_runner, kv_cache, logger)
        stale_template_id = f"sample-{sample_idx}-stale-template"

        try:
            warm_prompt_cache(engine, kv_cache, stale_template_id, stale_prompt)
            for recomp_ratio in ratios:
                qaw_session_id = f"sample-{sample_idx}-qaw-{recomp_ratio}"
                seed_session_cache(kv_cache, stale_template_id, qaw_session_id)
                qaw_req = GenerateRequest(
                    session_id=qaw_session_id,
                    prompt=final_prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    use_cache=True,
                    recompute_strategy="query_aware",
                    recomp_ratio=recomp_ratio,
                    suffix_len=args.suffix_len,
                    query_text=query_text,
                )
                qaw_res = engine.generate(qaw_req)

                qaw_ttft_by_ratio[recomp_ratio].append(qaw_res.first_token_latency_s)
                qaw_total_by_ratio[recomp_ratio].append(qaw_res.total_latency_s)
                if (qaw_res.recompute_mode == "query_aware_recompute"
                        and qaw_res.recomputed_tokens > 0):
                    qaw_true_recompute_by_ratio[recomp_ratio] += 1

                qaw_f1 = _safe_max_f1(qaw_res.generated_text, answers,
                                      model_runner.tokenizer)
                if qaw_f1 is not None:
                    qaw_f1_by_ratio[recomp_ratio].append(qaw_f1)
        finally:
            cleanup_runtime(kv_cache, engine)
            mem_after_cleanup = get_cuda_mem_mb()
            print(
                f"[curve] 样本 {processed_samples}/{sample_limit} 完成, "
                f"已清理显存: {format_cuda_mem(mem_after_cleanup)}",
                flush=True,
            )

    for group_idx, recomp_ratio in enumerate(ratios, start=1):
        output_writer.append_json({
            "event": "run_summary",
            "recomp_ratio": recomp_ratio,
            "sample_count": processed_samples,
            "full_reuse_avg_ttft_s": baseline_stats["full_reuse_avg_ttft_s"],
            "kv_diff_avg_ttft_s": baseline_stats["kv_diff_avg_ttft_s"],
            "query_aware_avg_ttft_s": _mean(qaw_ttft_by_ratio[recomp_ratio]),
            "full_prefill_avg_ttft_s": baseline_stats["full_prefill_avg_ttft_s"],
            "full_reuse_avg_total_s": baseline_stats["full_reuse_avg_total_s"],
            "kv_diff_avg_total_s": baseline_stats["kv_diff_avg_total_s"],
            "query_aware_avg_total_s": _mean(qaw_total_by_ratio[recomp_ratio]),
            "full_prefill_avg_total_s": baseline_stats["full_prefill_avg_total_s"],
            "full_reuse_avg_f1": baseline_stats["full_reuse_avg_f1"],
            "kv_diff_avg_f1": baseline_stats["kv_diff_avg_f1"],
            "query_aware_avg_f1": _mean(qaw_f1_by_ratio[recomp_ratio]),
            "full_prefill_avg_f1": baseline_stats["full_prefill_avg_f1"],
            "kvd_true_recompute_count": baseline_stats["kvd_true_recompute_count"],
            "qaw_true_recompute_count": qaw_true_recompute_by_ratio[recomp_ratio],
            "kvd_enabled": False,
            "ended_at": utc8_now_str(),
        })
        print(
            f"[curve] 组完成 {group_idx}/{total_groups}, "
            f"run_summary 已写入, recomp_ratio={recomp_ratio:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
