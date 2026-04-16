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


def seed_session_cache(kv_cache: KVCacheManager, src_session_id: str,
                       dst_session_id: str):
    """复制一个会话的 KV 到另一个会话。"""
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
    """将指定 prompt 做预填充并写入缓存会话（max_new_tokens=0）。"""
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


def build_stale_prompt(doc_prompts, query_text):
    """构造与真实请求不同的旧缓存 prompt，确保触发“前缀不匹配重算”分支。"""
    stale_query = f"请仅用于缓存预热的占位问题：{query_text.strip()}"
    return build_final_prompt(doc_prompts, stale_query)


def main() -> None:
    cfg = RuntimeConfig()
    cfg.max_new_tokens = 10
    kvd_ratio = 0.16
    kvd_suffix_len = 32
    qaw_ratio = 0.50
    qaw_suffix_len = 32

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
    logger.info(
        "本次实验设置: kv_diff(ratio=%.2f,suffix_len=%d), query_aware(ratio=%.2f,suffix_len=%d)",
        kvd_ratio,
        kvd_suffix_len,
        qaw_ratio,
        qaw_suffix_len,
    )
    output_writer.append_json({
        "event": "run_start",
        "script": "example/blend.py",
        "console_log_file": console_log_path,
        "experiment_mode": "true_recompute_comparison",
        "started_at": utc8_now_str(),
        "model": cfg.model_name,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "kv_diff_ratio": kvd_ratio,
        "kv_diff_suffix_len": kvd_suffix_len,
        "query_aware_ratio": qaw_ratio,
        "query_aware_suffix_len": qaw_suffix_len,
    })

    reuse_ttft_list = []
    reuse_total_list = []
    kvd_ttft_list = []
    qaw_ttft_list = []
    base_ttft_list = []
    reuse_reuse_tokens_list = []
    kvd_total_list = []
    qaw_total_list = []
    base_total_list = []
    kvd_true_recompute_count = 0
    qaw_true_recompute_count = 0

    for sample_idx in range(1, 11):
        input_path = os.path.join(ROOT_DIR, "inputs", f"{sample_idx}.json")
        with open(input_path, "r", encoding="utf-8") as f:
            ex = json.load(f)

        chunk_num = ex["chunk_num"]
        doc_prompts = [ex[str(i)] for i in range(chunk_num)]
        query_text = ex["query"]

        logger.info("\n===== 样本 %d 开始，chunk_num=%d =====", sample_idx, chunk_num)
        final_prompt = build_final_prompt(doc_prompts, query_text)
        stale_prompt = build_stale_prompt(doc_prompts, query_text)

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

        # 方法 2：full reuse（先对同一 prompt 预填充，再完整复用 KV）。
        reuse_template_id = f"sample-reuse-template-{sample_idx}"
        warm_prompt_cache(engine, kv_cache, reuse_template_id, final_prompt)
        reuse_session_id = f"sample-full-reuse-{sample_idx}"
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

        # 准备真实重算对比：先构造同一份“非前缀旧缓存”，分别喂给 kvd / qaw。
        stale_template_id = f"sample-stale-template-{sample_idx}"
        stale_warm_res = warm_prompt_cache(engine, kv_cache, stale_template_id,
                                           stale_prompt)

        # 方法 3：高 KV 偏差重算（从非前缀旧缓存启动）。
        kvd_session_id = f"sample-kvd-{sample_idx}"
        seed_session_cache(kv_cache, stale_template_id, kvd_session_id)
        kvd_req = GenerateRequest(
            session_id=kvd_session_id,
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=True,
            recompute_strategy="kv_diff",
            recomp_ratio=kvd_ratio,
            suffix_len=kvd_suffix_len,
            query_text=query_text,
        )
        kvd_res = engine.generate(kvd_req)

        # 方法 4：Query-aware 重算（从同一份非前缀旧缓存启动）。
        qaw_session_id = f"sample-qaw-{sample_idx}"
        seed_session_cache(kv_cache, stale_template_id, qaw_session_id)
        qaw_req = GenerateRequest(
            session_id=qaw_session_id,
            prompt=final_prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_cache=True,
            recompute_strategy="query_aware",
            recomp_ratio=qaw_ratio,
            suffix_len=qaw_suffix_len,
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

        logger.info(
            "sample=%d full_prefill(ttft=%.4f,total=%.4f) full_reuse(ttft=%.4f,total=%.4f,reused=%d,mode=%s) kvd(ttft=%.4f,total=%.4f,recomp=%d,mode=%s,true=%s) qaw(ttft=%.4f,total=%.4f,recomp=%d,mode=%s,true=%s)",
            sample_idx,
            base_res.first_token_latency_s,
            base_res.total_latency_s,
            reuse_res.first_token_latency_s,
            reuse_res.total_latency_s,
            reuse_res.reused_prefix_tokens,
            reuse_res.recompute_mode,
            kvd_res.first_token_latency_s,
            kvd_res.total_latency_s,
            kvd_res.recomputed_tokens,
            kvd_res.recompute_mode,
            str(kvd_true_recompute),
            qaw_res.first_token_latency_s,
            qaw_res.total_latency_s,
            qaw_res.recomputed_tokens,
            qaw_res.recompute_mode,
            str(qaw_true_recompute),
        )

        reuse_ttft_list.append(reuse_res.first_token_latency_s)
        kvd_ttft_list.append(kvd_res.first_token_latency_s)
        qaw_ttft_list.append(qaw_res.first_token_latency_s)
        base_ttft_list.append(base_res.first_token_latency_s)
        reuse_total_list.append(reuse_res.total_latency_s)
        reuse_reuse_tokens_list.append(reuse_res.reused_prefix_tokens)
        kvd_total_list.append(kvd_res.total_latency_s)
        qaw_total_list.append(qaw_res.total_latency_s)
        base_total_list.append(base_res.total_latency_s)

        output_writer.append_json({
            "event": "sample_result",
            "sample_idx": sample_idx,
            "chunk_num": chunk_num,
            "stale_cache_prompt_tokens": stale_warm_res.prompt_tokens,
            "kv_diff_ratio": kvd_ratio,
            "kv_diff_suffix_len": kvd_suffix_len,
            "query_aware_ratio": qaw_ratio,
            "query_aware_suffix_len": qaw_suffix_len,
            "full_prefill": {
                "generated_text": base_res.generated_text,
                "recompute_mode": base_res.recompute_mode,
                "ttft_s": base_res.first_token_latency_s,
                "total_s": base_res.total_latency_s,
            },
            "full_reuse": {
                "generated_text": reuse_res.generated_text,
                "recompute_mode": reuse_res.recompute_mode,
                "ttft_s": reuse_res.first_token_latency_s,
                "total_s": reuse_res.total_latency_s,
                "reused_prefix_tokens": reuse_res.reused_prefix_tokens,
            },
            "kv_diff": {
                "generated_text": kvd_res.generated_text,
                "recompute_mode": kvd_res.recompute_mode,
                "ttft_s": kvd_res.first_token_latency_s,
                "total_s": kvd_res.total_latency_s,
                "reused_prefix_tokens": kvd_res.reused_prefix_tokens,
                "recomputed_tokens": kvd_res.recomputed_tokens,
                "true_recompute": kvd_true_recompute,
            },
            "query_aware": {
                "generated_text": qaw_res.generated_text,
                "recompute_mode": qaw_res.recompute_mode,
                "ttft_s": qaw_res.first_token_latency_s,
                "total_s": qaw_res.total_latency_s,
                "reused_prefix_tokens": qaw_res.reused_prefix_tokens,
                "recomputed_tokens": qaw_res.recomputed_tokens,
                "true_recompute": qaw_true_recompute,
            },
        })

    output_writer.append_json({
        "event": "run_summary",
        "sample_count": len(kvd_ttft_list),
        "full_reuse_avg_ttft_s": (sum(reuse_ttft_list) / len(reuse_ttft_list))
        if reuse_ttft_list else None,
        "kv_diff_avg_ttft_s": (sum(kvd_ttft_list) / len(kvd_ttft_list))
        if kvd_ttft_list else None,
        "query_aware_avg_ttft_s": (sum(qaw_ttft_list) / len(qaw_ttft_list))
        if qaw_ttft_list else None,
        "full_prefill_avg_ttft_s": (sum(base_ttft_list) / len(base_ttft_list))
        if base_ttft_list else None,
        "full_reuse_avg_total_s": (sum(reuse_total_list) / len(reuse_total_list))
        if reuse_total_list else None,
        "kv_diff_avg_total_s": (sum(kvd_total_list) / len(kvd_total_list))
        if kvd_total_list else None,
        "query_aware_avg_total_s": (sum(qaw_total_list) / len(qaw_total_list))
        if qaw_total_list else None,
        "full_prefill_avg_total_s": (sum(base_total_list) / len(base_total_list))
        if base_total_list else None,
        "full_reuse_avg_reused_prefix_tokens":
        (sum(reuse_reuse_tokens_list) / len(reuse_reuse_tokens_list))
        if reuse_reuse_tokens_list else None,
        "kv_diff_ratio": kvd_ratio,
        "kv_diff_suffix_len": kvd_suffix_len,
        "query_aware_ratio": qaw_ratio,
        "query_aware_suffix_len": qaw_suffix_len,
        "kvd_true_recompute_count": kvd_true_recompute_count,
        "qaw_true_recompute_count": qaw_true_recompute_count,
        "ended_at": utc8_now_str(),
    })


if __name__ == "__main__":
    main()
