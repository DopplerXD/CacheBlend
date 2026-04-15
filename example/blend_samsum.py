"""SAMSum 实验脚本：四种策略对比（仅落盘日志，不向终端输出）。"""

import argparse
import importlib.util
import json
import os
import sys
from typing import List, Optional

# 允许从项目根目录导入模块（保持 `python example/blend_samsum.py` 可直接运行）。
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
        return max([compute_rl(pred_text, a) for a in answers])
    except Exception:
        return None


def build_prefix_prompt(doc_prompts: List[str]) -> str:
    return PREFIX_PROMPT + "".join(doc_prompts)


def build_final_prompt(doc_prompts: List[str], q_prompt: str) -> str:
    return build_prefix_prompt(doc_prompts) + q_prompt


def build_stale_prompt(doc_prompts: List[str]) -> str:
    stale_q_prompt = "\n\nDialogue: Placeholder cache-warm dialogue.\nSummary:"
    return build_final_prompt(doc_prompts, stale_q_prompt)


def seed_session_from_checkpoint(kv_cache: KVCacheManager, src_session_id: str,
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
    parser = argparse.ArgumentParser(description="SAMSum 对比实验脚本")
    parser.add_argument("--count", type=int, default=100, help="最多评测样本数")
    parser.add_argument("--qaw-ratio", type=float, default=0.7, help="query-aware 重算比例")
    parser.add_argument("--output-dir",
                        type=str,
                        default=os.path.join(ROOT_DIR, "outputs"),
                        help="实验日志输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = RuntimeConfig()
    cfg.max_new_tokens = 64

    # 按需求：脚本不向终端输出，统一写入 outputs/*.output。
    logger = setup_logger("blend_samsum", cfg.log_level)
    logger.disabled = True

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
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
    })

    dataset_path = os.path.join(ROOT_DIR, "inputs", "samsum.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    kvd_ttft_list: List[float] = []
    qaw_ttft_list: List[float] = []
    base_ttft_list: List[float] = []
    reuse_ttft_list: List[float] = []
    kvd_total_list: List[float] = []
    qaw_total_list: List[float] = []
    base_total_list: List[float] = []
    reuse_total_list: List[float] = []
    kvd_rl_list: List[float] = []
    qaw_rl_list: List[float] = []
    base_rl_list: List[float] = []
    reuse_rl_list: List[float] = []
    kvd_true_recompute_count = 0
    qaw_true_recompute_count = 0

    count = 0
    for sample_idx, ex in enumerate(eval_dataset, start=1):
        answers = list(ex.get("answers", []))
        doc_prompts, q_prompt = build_fewshot_prompt(ex)
        query_text = (ex.get("question", "") or "").strip()
        final_prompt = build_final_prompt(doc_prompts, q_prompt)
        stale_prompt = build_stale_prompt(doc_prompts)

        # 方法 1：基线（关闭缓存，完整 prefill）。
        base_req = GenerateRequest(
            session_id=f"samsum-baseline-{sample_idx}",
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=False,
            recompute_strategy="none",
        )
        base_res = engine.generate(base_req)

        # 方法 2：full reuse（先预填充同一 prompt，再完整复用 KV + 缓存 logits）。
        reuse_template_id = f"samsum-reuse-template-{sample_idx}"
        warm_prompt_cache(engine, kv_cache, reuse_template_id, final_prompt)
        reuse_session_id = f"samsum-reuse-{sample_idx}"
        seed_session_from_checkpoint(kv_cache, reuse_template_id, reuse_session_id)
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

        stale_template_id = f"samsum-stale-template-{sample_idx}"
        stale_warm_res = warm_prompt_cache(engine, kv_cache, stale_template_id,
                                           stale_prompt)

        # 方法 3：高 KV 偏差重算。
        kvd_session_id = f"samsum-kvd-{sample_idx}"
        seed_session_from_checkpoint(kv_cache, stale_template_id, kvd_session_id)
        kvd_req = GenerateRequest(
            session_id=kvd_session_id,
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=True,
            recompute_strategy="kv_diff",
            recomp_ratio=0.16,
            suffix_len=32,
            query_text=query_text,
        )
        kvd_res = engine.generate(kvd_req)

        # 方法 4：Query-aware 重算。
        qaw_session_id = f"samsum-qaw-{sample_idx}"
        seed_session_from_checkpoint(kv_cache, stale_template_id, qaw_session_id)
        qaw_req = GenerateRequest(
            session_id=qaw_session_id,
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=True,
            recompute_strategy="query_aware",
            recomp_ratio=args.qaw_ratio,
            suffix_len=32,
            query_text=query_text,
        )
        qaw_res = engine.generate(qaw_req)

        kvd_true_recompute = (kvd_res.recompute_mode == "kv_diff_recompute"
                              and kvd_res.recomputed_tokens > 0)
        qaw_true_recompute = (qaw_res.recompute_mode == "query_aware_recompute"
                              and qaw_res.recomputed_tokens > 0)
        if kvd_true_recompute:
            kvd_true_recompute_count += 1
        if qaw_true_recompute:
            qaw_true_recompute_count += 1

        reuse_ttft_list.append(reuse_res.first_token_latency_s)
        kvd_ttft_list.append(kvd_res.first_token_latency_s)
        qaw_ttft_list.append(qaw_res.first_token_latency_s)
        base_ttft_list.append(base_res.first_token_latency_s)
        reuse_total_list.append(reuse_res.total_latency_s)
        kvd_total_list.append(kvd_res.total_latency_s)
        qaw_total_list.append(qaw_res.total_latency_s)
        base_total_list.append(base_res.total_latency_s)

        reuse_rl = _safe_max_rougel(reuse_res.generated_text, answers)
        kvd_rl = _safe_max_rougel(kvd_res.generated_text, answers)
        qaw_rl = _safe_max_rougel(qaw_res.generated_text, answers)
        base_rl = _safe_max_rougel(base_res.generated_text, answers)
        if reuse_rl is not None:
            reuse_rl_list.append(reuse_rl)
        if kvd_rl is not None:
            kvd_rl_list.append(kvd_rl)
        if qaw_rl is not None:
            qaw_rl_list.append(qaw_rl)
        if base_rl is not None:
            base_rl_list.append(base_rl)

        output_writer.append_json({
            "event": "sample_result",
            "sample_idx": sample_idx,
            "chunk_num": len(doc_prompts),
            "question": ex.get("question", ""),
            "answers": answers,
            "stale_cache_prompt_tokens": stale_warm_res.prompt_tokens,
            "full_reuse": {
                "generated_text": reuse_res.generated_text,
                "ttft_s": reuse_res.first_token_latency_s,
                "total_s": reuse_res.total_latency_s,
                "reused_prefix_tokens": reuse_res.reused_prefix_tokens,
                "recompute_mode": reuse_res.recompute_mode,
                "rouge_l": reuse_rl,
            },
            "kv_diff": {
                "generated_text": kvd_res.generated_text,
                "ttft_s": kvd_res.first_token_latency_s,
                "total_s": kvd_res.total_latency_s,
                "reused_prefix_tokens": kvd_res.reused_prefix_tokens,
                "recomputed_tokens": kvd_res.recomputed_tokens,
                "recompute_mode": kvd_res.recompute_mode,
                "true_recompute": kvd_true_recompute,
                "rouge_l": kvd_rl,
            },
            "query_aware": {
                "generated_text": qaw_res.generated_text,
                "ttft_s": qaw_res.first_token_latency_s,
                "total_s": qaw_res.total_latency_s,
                "reused_prefix_tokens": qaw_res.reused_prefix_tokens,
                "recomputed_tokens": qaw_res.recomputed_tokens,
                "recompute_mode": qaw_res.recompute_mode,
                "true_recompute": qaw_true_recompute,
                "rouge_l": qaw_rl,
            },
            "full_prefill": {
                "generated_text": base_res.generated_text,
                "ttft_s": base_res.first_token_latency_s,
                "total_s": base_res.total_latency_s,
                "rouge_l": base_rl,
            },
        })

        count += 1
        if count >= args.count:
            break

    output_writer.append_json({
        "event": "run_summary",
        "sample_count": count,
        "full_reuse_avg_ttft_s": _mean(reuse_ttft_list),
        "kv_diff_avg_ttft_s": _mean(kvd_ttft_list),
        "query_aware_avg_ttft_s": _mean(qaw_ttft_list),
        "full_prefill_avg_ttft_s": _mean(base_ttft_list),
        "full_reuse_avg_total_s": _mean(reuse_total_list),
        "kv_diff_avg_total_s": _mean(kvd_total_list),
        "query_aware_avg_total_s": _mean(qaw_total_list),
        "full_prefill_avg_total_s": _mean(base_total_list),
        "full_reuse_avg_rouge_l": _mean(reuse_rl_list),
        "kv_diff_avg_rouge_l": _mean(kvd_rl_list),
        "query_aware_avg_rouge_l": _mean(qaw_rl_list),
        "full_prefill_avg_rouge_l": _mean(base_rl_list),
        "kvd_true_recompute_count": kvd_true_recompute_count,
        "qaw_true_recompute_count": qaw_true_recompute_count,
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
