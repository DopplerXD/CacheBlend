"""MusiQue 速度-质量曲线脚本：仅输出每个 recomp_ratio 的 run_summary。"""

import importlib.util
import json
import os
import sys
from typing import List, Optional

# 允许从项目根目录导入模块（保持 `python example/blend_musique_curve.py` 可直接运行）。
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
    "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words.\nPassages:\n"
)
QUERY_PROMPT = (
    "\n\nAnswer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"
)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


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
    kv_cache.put(dst_session_id, list(src.token_ids), src.past_key_values)
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


def ratio_grid() -> List[float]:
    # [0.15, 0.20, ..., 0.80] 共 14 组。
    return [round(0.15 + i * 0.05, 2) for i in range(14)]


def main() -> None:
    cfg = RuntimeConfig()
    cfg.max_new_tokens = 32

    # 默认每组最多测 200 条；当前按需求在 count=100 时提前停止。
    max_samples_per_ratio = 200
    stop_count = 30

    logger = setup_logger("blend_musique_curve", cfg.log_level)
    logger.disabled = True

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    output_writer = ExperimentOutputWriter.create(
        os.path.join(ROOT_DIR, "outputs"), run_tag="musique_curve")

    dataset_path = os.path.join(ROOT_DIR, "inputs", "musique_s.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    for recomp_ratio in ratio_grid():
        # 每组 ratio 重建 KV cache，避免跨组缓存污染。
        kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
        engine = InferenceEngine(model_runner, kv_cache, logger)

        reuse_ttft_list: List[float] = []
        kvd_ttft_list: List[float] = []
        qaw_ttft_list: List[float] = []
        base_ttft_list: List[float] = []
        reuse_total_list: List[float] = []
        kvd_total_list: List[float] = []
        qaw_total_list: List[float] = []
        base_total_list: List[float] = []
        reuse_f1_list: List[float] = []
        kvd_f1_list: List[float] = []
        qaw_f1_list: List[float] = []
        base_f1_list: List[float] = []
        kvd_true_recompute_count = 0
        qaw_true_recompute_count = 0

        count = 0
        for sample_idx, ex in enumerate(eval_dataset, start=1):
            if count >= max_samples_per_ratio or count >= stop_count:
                break
            count += 1

            answers = ex.get("answers", [])
            doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
            query_text = normalize_question(ex.get("question", ""))
            final_prompt = build_final_prompt(doc_prompts, q_prompt)
            stale_prompt = build_stale_prompt(doc_prompts, query_text)

            # 1) full reuse
            reuse_template_id = f"r{recomp_ratio}-reuse-template-{sample_idx}"
            warm_prompt_cache(engine, kv_cache, reuse_template_id, final_prompt)
            reuse_session_id = f"r{recomp_ratio}-reuse-{sample_idx}"
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

            # 为 kvd / qaw 准备同一份 stale cache。
            stale_template_id = f"r{recomp_ratio}-stale-template-{sample_idx}"
            warm_prompt_cache(engine, kv_cache, stale_template_id, stale_prompt)

            # 2) kv_diff
            kvd_session_id = f"r{recomp_ratio}-kvd-{sample_idx}"
            seed_session_cache(kv_cache, stale_template_id, kvd_session_id)
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

            # 3) query_aware（ratio sweep）
            qaw_session_id = f"r{recomp_ratio}-qaw-{sample_idx}"
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
                suffix_len=32,
                query_text=query_text,
            )
            qaw_res = engine.generate(qaw_req)

            # 4) full prefill
            base_req = GenerateRequest(
                session_id=f"r{recomp_ratio}-baseline-{sample_idx}",
                prompt=final_prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                use_cache=False,
                recompute_strategy="none",
            )
            base_res = engine.generate(base_req)

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

            reuse_f1 = max([compute_f1(reuse_res.generated_text, a, model_runner.tokenizer)
                            for a in answers]) if answers else None
            kvd_f1 = max([compute_f1(kvd_res.generated_text, a, model_runner.tokenizer)
                          for a in answers]) if answers else None
            qaw_f1 = max([compute_f1(qaw_res.generated_text, a, model_runner.tokenizer)
                          for a in answers]) if answers else None
            base_f1 = max([compute_f1(base_res.generated_text, a, model_runner.tokenizer)
                           for a in answers]) if answers else None
            if reuse_f1 is not None:
                reuse_f1_list.append(reuse_f1)
            if kvd_f1 is not None:
                kvd_f1_list.append(kvd_f1)
            if qaw_f1 is not None:
                qaw_f1_list.append(qaw_f1)
            if base_f1 is not None:
                base_f1_list.append(base_f1)

        # 每组 ratio 仅输出一条 run_summary，不输出 sample_result。
        output_writer.append_json({
            "event": "run_summary",
            "recomp_ratio": recomp_ratio,
            "sample_count": count,
            "full_reuse_avg_ttft_s": _mean(reuse_ttft_list),
            "kv_diff_avg_ttft_s": _mean(kvd_ttft_list),
            "query_aware_avg_ttft_s": _mean(qaw_ttft_list),
            "full_prefill_avg_ttft_s": _mean(base_ttft_list),
            "full_reuse_avg_total_s": _mean(reuse_total_list),
            "kv_diff_avg_total_s": _mean(kvd_total_list),
            "query_aware_avg_total_s": _mean(qaw_total_list),
            "full_prefill_avg_total_s": _mean(base_total_list),
            "full_reuse_avg_f1": _mean(reuse_f1_list),
            "kv_diff_avg_f1": _mean(kvd_f1_list),
            "query_aware_avg_f1": _mean(qaw_f1_list),
            "full_prefill_avg_f1": _mean(base_f1_list),
            "kvd_true_recompute_count": kvd_true_recompute_count,
            "qaw_true_recompute_count": qaw_true_recompute_count,
            "ended_at": utc8_now_str(),
        })


if __name__ == "__main__":
    main()
