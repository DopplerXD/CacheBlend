"""推理请求/响应的数据结构定义。"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerateRequest:
    """一次生成请求。"""

    session_id: str
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    use_cache: bool = True
    # 重算策略：none / kv_diff / query_aware
    recompute_strategy: str = "none"
    # 是否启用“高 KV 偏差 token 重算”策略（仅在存在旧会话缓存且前缀不匹配时生效）。
    enable_kv_diff_recompute: bool = False
    # 重算比例：在候选区间中按 KV 偏差分数选 Top-K。
    recomp_ratio: float = 0.16
    # 尾部强制重算 token 数（通常覆盖 query 尾段）。
    suffix_len: int = 16
    # query-aware 策略下可显式提供 query 文本。
    query_text: str = ""
    # query-aware 子变体：default / no_suffix / random_topk
    qaw_variant: str = "default"
    # random_topk 变体随机种子（<=0 时按 session_id 派生）。
    qaw_random_seed: int = 0


@dataclass
class GenerateResult:
    """一次生成结果（含关键统计信息）。"""

    generated_text: str
    full_text: str
    reused_prefix_tokens: int
    prompt_tokens: int
    generated_tokens: int
    total_latency_s: float
    first_token_latency_s: Optional[float]
    # 本次使用的预填充模式：full_prefill / cache_prefix_reuse / kv_diff_recompute / query_aware_recompute
    recompute_mode: str = "full_prefill"
    # 本次被选中重算的 token 数（kv_diff/query_aware 重算模式下有意义）。
    recomputed_tokens: int = 0
