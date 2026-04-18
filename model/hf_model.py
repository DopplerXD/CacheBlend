"""通用 HuggingFace causal LM 模型封装。"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    name = (dtype_name or "bfloat16").lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


class HFModelRunner:
    """对外提供 token 编解码与前向推理接口。"""

    def __init__(self, model_name: str, device: str, model_dtype: str, logger):
        self.logger = logger
        self.model_name = model_name

        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("请求使用 cuda，但当前不可用，自动回退到 cpu")
            device = "cpu"
        self.device = torch.device(device)

        torch_dtype = _resolve_dtype(model_dtype)
        if self.device.type == "cpu" and torch_dtype != torch.float32:
            # CPU 上 bf16/fp16 兼容性不稳定，MVP 默认用 fp32。
            torch_dtype = torch.float32

        self.logger.info("加载 tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       trust_remote_code=True)

        self.logger.info("加载模型: %s, device=%s, dtype=%s", model_name,
                         self.device, str(torch_dtype))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def encode_no_special(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @staticmethod
    def get_past_len(past_key_values: Any) -> int:
        """读取 past_key_values 中已缓存的序列长度。"""
        if past_key_values is None:
            return 0
        # HF causal LM 通常是：layer -> (k, v), k.shape=[bsz, heads, seq, dim]
        # 这里取 seq 维长度。
        return int(past_key_values[0][0].shape[-2])

    @staticmethod
    def truncate_past_key_values(past_key_values: Any, target_len: int) -> Any:
        """按序列长度裁剪 past_key_values。"""
        if past_key_values is None:
            return None
        if target_len < 0:
            raise ValueError("target_len 不能小于 0")

        truncated = []
        for layer in past_key_values:
            if len(layer) < 2:
                raise ValueError("past_key_values 的 layer 结构异常，至少应包含 (k, v)")
            k, v, *rest = layer
            truncated_layer = (k[:, :, :target_len, :], v[:, :, :target_len, :], *rest)
            truncated.append(truncated_layer)
        return tuple(truncated)

    @torch.inference_mode()
    def lookup_token_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """查询 token 的输入 embedding，返回形状 [T, H]。"""
        if len(token_ids) == 0:
            raise ValueError("token_ids 不能为空")
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        emb_layer = self.model.get_input_embeddings()
        return emb_layer(ids)

    @torch.inference_mode()
    def forward_tokens(self,
                       input_token_ids: List[int],
                       past_key_values: Any = None,
                       position_ids: Optional[List[int]] = None) -> Tuple[torch.Tensor, Any]:
        """执行一次前向，返回 logits 与新的 past_key_values。"""
        if len(input_token_ids) == 0:
            raise ValueError("input_token_ids 不能为空")

        input_ids = torch.tensor([input_token_ids],
                                 dtype=torch.long,
                                 device=self.device)
        past_len = self.get_past_len(past_key_values)
        attn_len = past_len + len(input_token_ids)
        attention_mask = torch.ones((1, attn_len),
                                    dtype=torch.long,
                                    device=self.device)
        position_tensor = None
        if position_ids is not None:
            if len(position_ids) != len(input_token_ids):
                raise ValueError("position_ids 长度需与 input_token_ids 一致")
            position_tensor = torch.tensor([position_ids],
                                           dtype=torch.long,
                                           device=self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_tensor,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, outputs.past_key_values
