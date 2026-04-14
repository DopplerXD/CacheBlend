"""WikiMQA 实验脚本：三种策略对比（仅落盘日志，不向终端输出）。"""

import importlib.util
import json
import os
import sys
from typing import List, Optional, Tuple

# 允许从项目根目录导入模块（保持 `python example/blend_wikimqa.py` 可直接运行）。
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


def _normalize_answers(raw_answers) -> List[str]:
    """将 WikiMQA 的嵌套答案结构统一成字符串列表。"""
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


def build_prefix_prompt(doc_prompts: List[str]) -> str:
    """构造文档前缀，作为 chunk checkpoint 的可复用前缀。"""
    return PREFIX_PROMPT + "".join(doc_prompts)


def build_final_prompt(doc_prompts: List[str], q_prompt: str) -> str:
    """构造最终问答 prompt。"""
    return build_prefix_prompt(doc_prompts) + q_prompt


def warm_chunk_checkpoints(engine: InferenceEngine, kv_cache: KVCacheManager,
                           base_session_id: str,
                           doc_prompts: List[str]) -> List[str]:
    """按 chunk 边界建立 checkpoint：ckpt-1, ckpt-2, ..."""
    checkpoint_ids = []
    for i in range(1, len(doc_prompts) + 1):
        checkpoint_id = f"{base_session_id}-ckpt-{i}"
        kv_cache.clear(checkpoint_id)
        checkpoint_prompt = build_prefix_prompt(doc_prompts[:i])
        warm_req = GenerateRequest(
            session_id=checkpoint_id,
            prompt=checkpoint_prompt,
            max_new_tokens=0,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            recompute_strategy="none",
        )
        engine.generate(warm_req)
        checkpoint_ids.append(checkpoint_id)
    return checkpoint_ids


def select_best_checkpoint(kv_cache: KVCacheManager, model_runner: YiModelRunner,
                           checkpoint_ids: List[str],
                           target_prompt: str) -> Tuple[Optional[str], int]:
    """从 checkpoints 中选最长前缀匹配项。"""
    target_tokens = model_runner.encode(target_prompt)
    best_id = None
    best_len = -1
    for sid in checkpoint_ids:
        entry = kv_cache.get(sid)
        if entry is None:
            continue
        cur_len = len(entry.token_ids)
        if cur_len <= len(target_tokens) and target_tokens[:cur_len] == entry.token_ids:
            if cur_len > best_len:
                best_id = sid
                best_len = cur_len
    return best_id, best_len


def seed_session_from_checkpoint(kv_cache: KVCacheManager, src_session_id: str,
                                 dst_session_id: str) -> bool:
    """将 checkpoint KV 复制到运行 session。"""
    src = kv_cache.get(src_session_id)
    if src is None:
        return False
    kv_cache.clear(dst_session_id)
    kv_cache.put(dst_session_id, list(src.token_ids), src.past_key_values)
    return True


def main() -> None:
    cfg = RuntimeConfig()
    cfg.max_new_tokens = 32

    # 按需求：脚本不向终端输出，统一写入 outputs/*.output。
    logger = setup_logger("blend_wikimqa", cfg.log_level)
    logger.disabled = True

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
    engine = InferenceEngine(model_runner, kv_cache, logger)

    output_writer = ExperimentOutputWriter.create(
        os.path.join(ROOT_DIR, "outputs"), run_tag="wikimqa")
    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend_wikimqa.py",
        "dataset": "inputs/wikimqa_s.json",
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
    })

    dataset_path = os.path.join(ROOT_DIR, "inputs", "wikimqa_s.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    kvd_ttft_list: List[float] = []
    qaw_ttft_list: List[float] = []
    base_ttft_list: List[float] = []
    kvd_total_list: List[float] = []
    qaw_total_list: List[float] = []
    base_total_list: List[float] = []
    kvd_f1_list: List[float] = []
    qaw_f1_list: List[float] = []
    base_f1_list: List[float] = []

    count = 0
    for sample_idx, ex in enumerate(eval_dataset, start=1):
        count += 1
        answer_texts = _normalize_answers(ex.get("answers", []))
        doc_prompts, q_prompt = build_qa_prompt(ex, QUERY_PROMPT)
        query_text = normalize_question(ex.get("question", ""))
        final_prompt = build_final_prompt(doc_prompts, q_prompt)

        checkpoint_base = f"wikimqa-ckpt-{sample_idx}"
        checkpoint_ids = warm_chunk_checkpoints(engine, kv_cache, checkpoint_base,
                                                doc_prompts)
        best_ckpt_id, best_ckpt_len = select_best_checkpoint(
            kv_cache=kv_cache,
            model_runner=model_runner,
            checkpoint_ids=checkpoint_ids,
            target_prompt=final_prompt,
        )

        # 方法 1：高 KV 偏差重算。
        kvd_session_id = f"wikimqa-kvd-{sample_idx}"
        if best_ckpt_id is not None:
            seed_session_from_checkpoint(kv_cache, best_ckpt_id, kvd_session_id)
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

        # 方法 2：Query-aware 重算。
        qaw_session_id = f"wikimqa-qaw-{sample_idx}"
        if best_ckpt_id is not None:
            seed_session_from_checkpoint(kv_cache, best_ckpt_id, qaw_session_id)
        qaw_req = GenerateRequest(
            session_id=qaw_session_id,
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=True,
            recompute_strategy="query_aware",
            recomp_ratio=0.3,
            suffix_len=32,
            query_text=query_text,
        )
        qaw_res = engine.generate(qaw_req)

        # 方法 3：基线（关闭缓存，完整 prefill）。
        base_req = GenerateRequest(
            session_id=f"wikimqa-baseline-{sample_idx}",
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=False,
            recompute_strategy="none",
        )
        base_res = engine.generate(base_req)

        kvd_ttft_list.append(kvd_res.first_token_latency_s)
        qaw_ttft_list.append(qaw_res.first_token_latency_s)
        base_ttft_list.append(base_res.first_token_latency_s)
        kvd_total_list.append(kvd_res.total_latency_s)
        qaw_total_list.append(qaw_res.total_latency_s)
        base_total_list.append(base_res.total_latency_s)

        kvd_f1 = max([compute_f1(kvd_res.generated_text, a, model_runner.tokenizer)
                      for a in answer_texts]) if answer_texts else None
        qaw_f1 = max([compute_f1(qaw_res.generated_text, a, model_runner.tokenizer)
                      for a in answer_texts]) if answer_texts else None
        base_f1 = max([compute_f1(base_res.generated_text, a, model_runner.tokenizer)
                       for a in answer_texts]) if answer_texts else None
        if kvd_f1 is not None:
            kvd_f1_list.append(kvd_f1)
        if qaw_f1 is not None:
            qaw_f1_list.append(qaw_f1)
        if base_f1 is not None:
            base_f1_list.append(base_f1)

        output_writer.append_json({
            "event": "sample_result",
            "sample_idx": sample_idx,
            "chunk_num": len(doc_prompts),
            "question": ex.get("question", ""),
            "answers": answer_texts,
            "selected_checkpoint": best_ckpt_id,
            "selected_prefix_tokens": best_ckpt_len,
            "kv_diff": {
                "generated_text": kvd_res.generated_text,
                "ttft_s": kvd_res.first_token_latency_s,
                "total_s": kvd_res.total_latency_s,
                "reused_prefix_tokens": kvd_res.reused_prefix_tokens,
                "recomputed_tokens": kvd_res.recomputed_tokens,
                "f1": kvd_f1,
            },
            "query_aware": {
                "generated_text": qaw_res.generated_text,
                "ttft_s": qaw_res.first_token_latency_s,
                "total_s": qaw_res.total_latency_s,
                "reused_prefix_tokens": qaw_res.reused_prefix_tokens,
                "recomputed_tokens": qaw_res.recomputed_tokens,
                "f1": qaw_f1,
            },
            "full_prefill": {
                "generated_text": base_res.generated_text,
                "ttft_s": base_res.first_token_latency_s,
                "total_s": base_res.total_latency_s,
                "f1": base_f1,
            },
        })
        if count == 10:
            break

    output_writer.append_json({
        "event": "run_summary",
        # "sample_count": len(eval_dataset),
        "sample_count": count,
        "kv_diff_avg_ttft_s": _mean(kvd_ttft_list),
        "query_aware_avg_ttft_s": _mean(qaw_ttft_list),
        "full_prefill_avg_ttft_s": _mean(base_ttft_list),
        "kv_diff_avg_total_s": _mean(kvd_total_list),
        "query_aware_avg_total_s": _mean(qaw_total_list),
        "full_prefill_avg_total_s": _mean(base_total_list),
        "kv_diff_avg_f1": _mean(kvd_f1_list),
        "query_aware_avg_f1": _mean(qaw_f1_list),
        "full_prefill_avg_f1": _mean(base_f1_list),
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
