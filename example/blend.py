"""MVP 示例：按旧 blend.py 链路演示“缓存路径 vs 全量 prefill”。"""

import json
import logging
import os
import sys

# 允许从项目根目录导入模块（保持 `python example/blend.py` 可直接运行）。
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cache.kv_cache import KVCacheManager
from config import RuntimeConfig
from engine.inference_engine import InferenceEngine
from model.yi_model import YiModelRunner
from schema.types import GenerateRequest
from utils.experiment_output import ExperimentOutputWriter, utc8_now_str
from utils.logging_utils import setup_logger


def _format_docs(doc_prompts):
    docs = []
    for i, doc in enumerate(doc_prompts, start=1):
        docs.append(f"[文档{i}]\n{doc.strip()}")
    return "\n\n".join(docs)


def build_prefix_prompt(doc_prompts):
    """构造“文档前缀”Prompt（按 chunk 边界 checkpoint 用）。

    设计目标：
    1. 该字符串必须是最终问答 prompt 的严格前缀。
    2. 允许在 chunk 边界建立可复用 checkpoint。
    """
    docs_text = _format_docs(doc_prompts)
    return (
        "你是一个问答助手。请基于给定文档回答问题。\n\n"
        f"文档内容:\n{docs_text}\n\n"
        "问题:\n"
    )


def build_final_prompt(doc_prompts, query_text):
    """构造最终问答 Prompt。"""
    return build_prefix_prompt(doc_prompts) + f"{query_text.strip()}\n\n回答:"


def warm_chunk_checkpoints(engine: InferenceEngine, kv_cache: KVCacheManager,
                           base_session_id: str, doc_prompts):
    """按 chunk 边界做 checkpoint：ckpt-1, ckpt-2, ..."""
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
                           checkpoint_ids, target_prompt):
    """从 chunk 边界 checkpoints 中选择最长前缀匹配项。"""
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
                                 dst_session_id: str):
    """将 checkpoint KV 复制到运行 session。"""
    src = kv_cache.get(src_session_id)
    if src is None:
        return False
    kv_cache.clear(dst_session_id)
    kv_cache.put(dst_session_id, list(src.token_ids), src.past_key_values)
    return True


def main() -> None:
    cfg = RuntimeConfig()
    cfg.max_new_tokens = 10

    output_writer = ExperimentOutputWriter.create(
        os.path.join(ROOT_DIR, "outputs"), run_tag="blend")
    console_log_path = output_writer.file_path.replace(".output",
                                                        "_console.output")
    logger = setup_logger("blend_mvp", cfg.log_level)
    # 将“控制台风格日志”改为写入文件，不输出到终端。
    for h in list(logger.handlers):
        logger.removeHandler(h)
    file_handler = logging.FileHandler(console_log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.propagate = False

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
    engine = InferenceEngine(model_runner, kv_cache, logger)
    logger.info("实验结构化日志文件: %s", output_writer.file_path)
    logger.info("实验控制台风格日志文件: %s", console_log_path)
    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend.py",
        "console_log_file": console_log_path,
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
    })

    kvd_ttft_list = []
    qaw_ttft_list = []
    base_ttft_list = []
    kvd_total_list = []
    qaw_total_list = []
    base_total_list = []

    for sample_idx in range(1, 11):
        input_path = os.path.join(ROOT_DIR, "inputs", f"{sample_idx}.json")
        with open(input_path, "r", encoding="utf-8") as f:
            ex = json.load(f)

        chunk_num = ex["chunk_num"]
        doc_prompts = [ex[str(i)] for i in range(chunk_num)]
        query_text = ex["query"]

        logger.info("\n===== 样本 %d 开始，chunk_num=%d =====", sample_idx, chunk_num)
        final_prompt = build_final_prompt(doc_prompts, query_text)

        # 先建立一套“按 chunk 边界”的 checkpoint 缓存。
        checkpoint_base = f"sample-ckpt-{sample_idx}"
        checkpoint_ids = warm_chunk_checkpoints(engine, kv_cache, checkpoint_base,
                                                doc_prompts)
        best_ckpt_id, best_ckpt_len = select_best_checkpoint(
            kv_cache=kv_cache,
            model_runner=model_runner,
            checkpoint_ids=checkpoint_ids,
            target_prompt=final_prompt,
        )
        logger.info("sample=%d checkpoint=%s prefix_tokens=%d", sample_idx,
                    best_ckpt_id, best_ckpt_len)

        # 方法 1：基线（关闭缓存，完整 prefill）。
        base_req = GenerateRequest(
            session_id=f"baseline-{sample_idx}",
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=False,
            recompute_strategy="none",
        )
        base_res = engine.generate(base_req)

        # 方法 2：高 KV 偏差重算
        kvd_session_id = f"sample-kvd-{sample_idx}"
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

        # 方法 3：Query-aware 重算
        qaw_session_id = f"sample-qaw-{sample_idx}"
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
            recomp_ratio=0.16,
            suffix_len=32,
            query_text=query_text,
        )
        qaw_res = engine.generate(qaw_req)

        logger.info(
            "sample=%d full(ttft=%.4f,total=%.4f) kvd(ttft=%.4f,total=%.4f,recomp=%d,mode=%s) qaw(ttft=%.4f,total=%.4f,recomp=%d,mode=%s)",
            sample_idx,
            base_res.first_token_latency_s,
            base_res.total_latency_s,
            kvd_res.first_token_latency_s,
            kvd_res.total_latency_s,
            kvd_res.recomputed_tokens,
            kvd_res.recompute_mode,
            qaw_res.first_token_latency_s,
            qaw_res.total_latency_s,
            qaw_res.recomputed_tokens,
            qaw_res.recompute_mode,
        )

        kvd_ttft_list.append(kvd_res.first_token_latency_s)
        qaw_ttft_list.append(qaw_res.first_token_latency_s)
        base_ttft_list.append(base_res.first_token_latency_s)
        kvd_total_list.append(kvd_res.total_latency_s)
        qaw_total_list.append(qaw_res.total_latency_s)
        base_total_list.append(base_res.total_latency_s)

        output_writer.append_json({
            "event": "sample_result",
            "sample_idx": sample_idx,
            "chunk_num": chunk_num,
            "selected_checkpoint": best_ckpt_id,
            "selected_prefix_tokens": best_ckpt_len,
            "full_prefill": {
                "generated_text": base_res.generated_text,
                "ttft_s": base_res.first_token_latency_s,
                "total_s": base_res.total_latency_s,
            },
            "kv_diff": {
                "generated_text": kvd_res.generated_text,
                "recompute_mode": kvd_res.recompute_mode,
                "ttft_s": kvd_res.first_token_latency_s,
                "total_s": kvd_res.total_latency_s,
                "reused_prefix_tokens": kvd_res.reused_prefix_tokens,
                "recomputed_tokens": kvd_res.recomputed_tokens,
            },
            "query_aware": {
                "generated_text": qaw_res.generated_text,
                "recompute_mode": qaw_res.recompute_mode,
                "ttft_s": qaw_res.first_token_latency_s,
                "total_s": qaw_res.total_latency_s,
                "reused_prefix_tokens": qaw_res.reused_prefix_tokens,
                "recomputed_tokens": qaw_res.recomputed_tokens,
            },
        })

    output_writer.append_json({
        "event": "run_summary",
        "sample_count": len(kvd_ttft_list),
        "kv_diff_avg_ttft_s": (sum(kvd_ttft_list) / len(kvd_ttft_list))
        if kvd_ttft_list else None,
        "query_aware_avg_ttft_s": (sum(qaw_ttft_list) / len(qaw_ttft_list))
        if qaw_ttft_list else None,
        "full_prefill_avg_ttft_s": (sum(base_ttft_list) / len(base_ttft_list))
        if base_ttft_list else None,
        "kv_diff_avg_total_s": (sum(kvd_total_list) / len(kvd_total_list))
        if kvd_total_list else None,
        "query_aware_avg_total_s": (sum(qaw_total_list) / len(qaw_total_list))
        if qaw_total_list else None,
        "full_prefill_avg_total_s": (sum(base_total_list) / len(base_total_list))
        if base_total_list else None,
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
