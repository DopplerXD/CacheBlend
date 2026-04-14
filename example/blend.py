"""MVP 示例：按旧 blend.py 链路演示“缓存路径 vs 全量 prefill”。"""

import json
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
from utils.logging_utils import setup_logger


def build_prompt(doc_prompts, query_text):
    """构造统一 Prompt。

    设计目标：
    1. 让“分块暖缓存”与“最终问答”使用同一模板。
    2. 便于后续接入 query-aware token selection。
    """
    docs = []
    for i, doc in enumerate(doc_prompts, start=1):
        docs.append(f"[文档{i}]\n{doc.strip()}")
    docs_text = "\n\n".join(docs)
    return (
        "你是一个问答助手。请基于给定文档回答问题。\n\n"
        f"文档内容:\n{docs_text}\n\n"
        f"问题:\n{query_text.strip()}\n\n"
        "回答:"
    )


def warm_session_cache(engine: InferenceEngine, kv_cache: KVCacheManager,
                       session_id: str, doc_prompts):
    """按 chunk 顺序做纯 prefill，建立会话缓存。"""
    kv_cache.clear(session_id)
    warm_docs = []
    for chunk in doc_prompts:
        warm_docs.append(chunk)
        warm_prompt = build_prompt(warm_docs, "")
        warm_req = GenerateRequest(
            session_id=session_id,
            prompt=warm_prompt,
            max_new_tokens=0,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            recompute_strategy="none",
        )
        engine.generate(warm_req)


def main() -> None:
    cfg = RuntimeConfig()
    cfg.max_new_tokens = 10

    logger = setup_logger("blend_mvp", cfg.log_level)
    logger.info("启动 MVP 示例，模型=%s", cfg.model_name)

    model_runner = YiModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger)
    engine = InferenceEngine(model_runner, kv_cache, logger)

    for sample_idx in range(1, 11):
        input_path = os.path.join(ROOT_DIR, "inputs", f"{sample_idx}.json")
        with open(input_path, "r", encoding="utf-8") as f:
            ex = json.load(f)

        chunk_num = ex["chunk_num"]
        doc_prompts = [ex[str(i)] for i in range(chunk_num)]
        query_text = ex["query"]

        logger.info("\n===== 样本 %d 开始，chunk_num=%d =====", sample_idx, chunk_num)
        final_prompt = build_prompt(doc_prompts, query_text)

        # 方法 1：高 KV 偏差重算
        kvd_session_id = f"sample-kvd-{sample_idx}"
        warm_session_cache(engine, kv_cache, kvd_session_id, doc_prompts)
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

        # 方法 2：Query-aware 重算
        qaw_session_id = f"sample-qaw-{sample_idx}"
        warm_session_cache(engine, kv_cache, qaw_session_id, doc_prompts)
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

        # 方法 3：基线（关闭缓存，完整 prefill）。
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

        print(f"[Sample {sample_idx}] KV-diff generation: {kvd_res.generated_text}")
        print(
            f"[Sample {sample_idx}] KV-diff TTFT: {kvd_res.first_token_latency_s:.4f}s | "
            f"Total: {kvd_res.total_latency_s:.4f}s | Reused: {kvd_res.reused_prefix_tokens} | "
            f"Mode: {kvd_res.recompute_mode} | Recomputed: {kvd_res.recomputed_tokens}"
        )
        print(f"[Sample {sample_idx}] Query-aware generation: {qaw_res.generated_text}")
        print(
            f"[Sample {sample_idx}] Query-aware TTFT: {qaw_res.first_token_latency_s:.4f}s | "
            f"Total: {qaw_res.total_latency_s:.4f}s | Reused: {qaw_res.reused_prefix_tokens} | "
            f"Mode: {qaw_res.recompute_mode} | Recomputed: {qaw_res.recomputed_tokens}"
        )
        print(f"[Sample {sample_idx}] Normal generation: {base_res.generated_text}")
        print(
            f"[Sample {sample_idx}] TTFT full prefill: {base_res.first_token_latency_s:.4f}s | "
            f"Total: {base_res.total_latency_s:.4f}s"
        )
        print("------------")


if __name__ == "__main__":
    main()
