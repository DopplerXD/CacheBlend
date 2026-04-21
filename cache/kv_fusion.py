"""KV 拼接与融合工具。"""

from __future__ import annotations

from typing import Any

import torch


def concat_past_key_values(left: Any, right: Any) -> Any:
    """按序列维拼接两段 past_key_values。"""
    if left is None:
        return clone_past_key_values(right)
    if right is None:
        return clone_past_key_values(left)

    merged = []
    for left_layer, right_layer in zip(left, right):
        left_k, left_v = left_layer[0], left_layer[1]
        right_k, right_v = right_layer[0], right_layer[1]
        merged_k = torch.cat([left_k, right_k], dim=-2)
        merged_v = torch.cat([left_v, right_v], dim=-2)
        merged.append((merged_k, merged_v, *left_layer[2:]))
    return tuple(merged)


def clone_past_key_values(past_key_values: Any) -> Any:
    """浅结构、深 tensor clone，避免修改缓存池中的原始 KV。"""
    if past_key_values is None:
        return None
    cloned = []
    for layer in past_key_values:
        k, v, *rest = layer
        cloned.append((k.clone(), v.clone(), *rest))
    return tuple(cloned)


def slice_past_key_values(past_key_values: Any, start_idx: int, end_idx: int) -> Any:
    """按序列维切出 [start_idx:end_idx) 的 past_key_values。"""
    if past_key_values is None:
        return None
    sliced = []
    for layer in past_key_values:
        k, v, *rest = layer
        sliced.append((k[:, :, start_idx:end_idx, :].clone(),
                       v[:, :, start_idx:end_idx, :].clone(), *rest))
    return tuple(sliced)


def scatter_selected_past_key_values(base_past_key_values: Any,
                                     selected_past_key_values: Any,
                                     selected_indices: torch.Tensor) -> Any:
    """将 packed selected KV 写回 base 的指定序列位置。"""
    if selected_past_key_values is None or selected_indices.numel() == 0:
        return base_past_key_values

    selected_num = int(selected_indices.numel())
    blended = []
    for base_layer, selected_layer in zip(base_past_key_values,
                                          selected_past_key_values):
        base_k, base_v = base_layer[0].clone(), base_layer[1].clone()
        selected_k, selected_v = selected_layer[0], selected_layer[1]
        base_k[:, :, selected_indices, :] = selected_k[:, :, :selected_num, :]
        base_v[:, :, selected_indices, :] = selected_v[:, :, :selected_num, :]
        blended.append((base_k, base_v, *base_layer[2:]))
    return tuple(blended)
