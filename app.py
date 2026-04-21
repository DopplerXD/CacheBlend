"""命令行入口：单次请求调用 MVP 引擎。"""

import argparse

from cache.kv_cache import KVCacheManager
from config import RuntimeConfig
from engine.inference_engine import InferenceEngine
from model.hf_model import HFModelRunner
from schema.types import GenerateRequest
from utils.logging_utils import setup_logger


def build_engine(cfg: RuntimeConfig):
    logger = setup_logger("app", cfg.log_level)
    model_runner = HFModelRunner(cfg.model_name, cfg.device, cfg.model_dtype,
                                 logger)
    kv_cache = KVCacheManager(cfg.kv_max_sessions, cfg.kv_ttl_seconds, logger, cfg.kv_max_chunks)
    engine = InferenceEngine(model_runner, kv_cache, logger)
    return engine, logger


def main() -> None:
    parser = argparse.ArgumentParser(description="HF Causal LM KV Cache MVP")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--session-id", type=str, default="demo-session")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--recompute-strategy",
                        type=str,
                        default="none",
                        choices=["none", "kv_diff", "query_aware"])
    parser.add_argument("--query-text",
                        type=str,
                        default="",
                        help="query-aware 策略下可单独指定 query 文本")
    parser.add_argument("--kv-diff-recompute", action="store_true")
    parser.add_argument("--recomp-ratio", type=float, default=0.16)
    parser.add_argument("--suffix-len", type=int, default=16)
    parser.add_argument("--qaw-type",
                        type=str,
                        default="packed",
                        choices=["packed", "window"])
    args = parser.parse_args()

    cfg = RuntimeConfig()
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens

    engine, logger = build_engine(cfg)

    req = GenerateRequest(
        session_id=args.session_id,
        prompt=args.prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        use_cache=not args.no_cache,
        recompute_strategy=args.recompute_strategy,
        enable_kv_diff_recompute=args.kv_diff_recompute,
        recomp_ratio=args.recomp_ratio,
        suffix_len=args.suffix_len,
        qaw_type=args.qaw_type,
        query_text=args.query_text,
    )

    res = engine.generate(req)
    logger.info("生成结果: %s", res.generated_text)
    logger.info(
        "统计: mode=%s, recomputed_tokens=%d, reused_prefix=%d, prompt_tokens=%d, generated_tokens=%d, ttft=%.4fs, total=%.4fs",
        res.recompute_mode,
        res.recomputed_tokens,
        res.reused_prefix_tokens,
        res.prompt_tokens,
        res.generated_tokens,
        res.first_token_latency_s or -1.0,
        res.total_latency_s,
    )


if __name__ == "__main__":
    main()
