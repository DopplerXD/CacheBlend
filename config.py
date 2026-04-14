"""运行配置。"""

from dataclasses import dataclass
import os


@dataclass
class RuntimeConfig:
    """MVP 运行参数。

    这里保留最少字段，先保证主链路可跑通。
    """

    # 目标模型，默认 Yi-6B。
    model_name: str = os.getenv("MODEL_NAME", "01-ai/Yi-6B")
    # 推理设备：cuda / cpu。
    device: str = os.getenv("DEVICE", "cuda")
    # 模型权重 dtype：bfloat16 / float16 / float32。
    model_dtype: str = os.getenv("MODEL_DTYPE", "bfloat16")

    # 生成参数（MVP 默认贪心）。
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "16"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("TOP_P", "1.0"))

    # KV 缓存策略（简单 LRU + TTL）。
    kv_max_sessions: int = int(os.getenv("KV_MAX_SESSIONS", "16"))
    kv_ttl_seconds: int = int(os.getenv("KV_TTL_SECONDS", "3600"))

    # 日志级别。
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
