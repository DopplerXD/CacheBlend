"""MVP 推理主链路：prefill + decode + 会话 KV 复用。"""

from __future__ import annotations

import time
import hashlib
from typing import Any, List, Tuple

import torch

from schema.types import GenerateRequest, GenerateResult


class InferenceEngine:
    """最小可用推理引擎。

    说明：
    1. 当前仅做会话级 KV 复用。
    2. 不做并发调度与工程化优化。
    3. 支持 kv_diff 与 query-aware 两种非前缀旧缓存下的选择性重算实验路径。
    """

    def __init__(self, model_runner, kv_cache_manager, logger):
        self.model_runner = model_runner
        self.kv_cache = kv_cache_manager
        self.logger = logger

    @staticmethod
    def _is_prefix(prefix_tokens: List[int], full_tokens: List[int]) -> bool:
        if len(prefix_tokens) > len(full_tokens):
            return False
        return full_tokens[:len(prefix_tokens)] == prefix_tokens

    def _sample_next_token(self, logits: torch.Tensor, temperature: float,
                           top_p: float) -> int:
        """从最后一个位置的 logits 采样下一个 token。

        MVP 策略：
        1. `temperature <= 0` 时使用 greedy。
        2. 否则使用 top-p 采样。
        """
        last_logits = logits[:, -1, :]

        if temperature <= 0:
            return int(torch.argmax(last_logits, dim=-1).item())

        scaled = last_logits / max(temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1).squeeze(0)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            cutoff = cumulative > top_p
            if torch.any(cutoff):
                first_cut = int(torch.argmax(cutoff.int()).item())
                sorted_probs[first_cut + 1:] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
            next_idx = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[next_idx].item())

        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    def _compute_kv_diff_scores(self, old_past_key_values: Any,
                                new_past_key_values: Any,
                                overlap_len: int) -> torch.Tensor:
        """计算 token 级 KV 偏差分数（使用 V 的 L2 差异并按层累加）。"""
        score = None
        for old_layer, new_layer in zip(old_past_key_values, new_past_key_values):
            old_v = old_layer[1][:, :, :overlap_len, :]
            new_v = new_layer[1][:, :, :overlap_len, :]
            layer_score = (new_v - old_v).pow(2).sum(dim=(0, 1, 3))
            score = layer_score if score is None else (score + layer_score)
        return score

    def _compute_query_aware_scores(self, prompt_token_ids: List[int],
                                    query_token_ids: List[int],
                                    overlap_len: int) -> torch.Tensor:
        """计算 query-aware token 相关性分数（embedding cosine）。"""
        if overlap_len <= 0:
            return torch.zeros(0, device=self.model_runner.device)

        if len(query_token_ids) == 0:
            # 若未提供 query，回退为 0 分向量（由尾部强制重算兜底）。
            return torch.zeros(overlap_len, device=self.model_runner.device)

        cand_emb = self.model_runner.lookup_token_embeddings(
            prompt_token_ids[:overlap_len])
        query_emb = self.model_runner.lookup_token_embeddings(query_token_ids)
        cand_emb = torch.nn.functional.normalize(cand_emb, dim=-1)
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        sim = cand_emb @ query_emb.transpose(0, 1)
        return sim.max(dim=1).values

    def _select_token_indices_from_scores(self,
                                          old_tokens: List[int],
                                          new_tokens: List[int],
                                          scores: torch.Tensor,
                                          recomp_ratio: float,
                                          suffix_len: int,
                                          force_changed: bool,
                                          random_topk: bool = False,
                                          random_seed: int = 0) -> torch.Tensor:
        """通用 token 选择器：Top-K(score) + 尾部强制 + 新增 token（可选强制 changed）。"""
        new_len = len(new_tokens)
        overlap_len = min(len(old_tokens), new_len, int(scores.shape[0]))
        force_start = max(0, new_len - max(suffix_len, 0))

        changed = {i for i in range(overlap_len) if old_tokens[i] != new_tokens[i]}
        added = set(range(overlap_len, new_len))
        forced = set(range(force_start, new_len))

        # Top-K 候选区间排除尾部强制重算部分。
        candidate_end = min(overlap_len, force_start)
        topk = set()
        if candidate_end > 0 and recomp_ratio > 0:
            topk_num = max(1, int(candidate_end * recomp_ratio))
            topk_num = min(topk_num, candidate_end)
            if random_topk:
                generator = torch.Generator(device=scores.device)
                generator.manual_seed(int(random_seed))
                top_indices = torch.randperm(
                    candidate_end, device=scores.device,
                    generator=generator)[:topk_num]
            else:
                top_indices = torch.topk(scores[:candidate_end], k=topk_num).indices
            topk = set(int(i) for i in top_indices.tolist())

        if force_changed:
            selected = sorted(changed | added | forced | topk)
        else:
            selected = sorted(added | forced | topk)
        if not selected:
            # 理论上至少会有 forced 或 added；这里兜底避免空索引。
            selected = [new_len - 1] if new_len > 0 else []
        return torch.tensor(selected, dtype=torch.long, device=scores.device)

    @staticmethod
    def _derive_stable_seed(session_id: str, fallback_seed: int) -> int:
        """派生稳定随机种子：优先使用显式 seed，否则由 session_id 生成。"""
        if fallback_seed > 0:
            return int(fallback_seed)
        digest = hashlib.sha256(session_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False)

    def _build_blended_past_key_values(self, old_past_key_values: Any,
                                       new_past_key_values: Any, new_len: int,
                                       selected_indices: torch.Tensor) -> Any:
        """构建混合 KV：未选中位置复用 old，选中位置替换为 new。"""
        blended = []
        for old_layer, new_layer in zip(old_past_key_values, new_past_key_values):
            old_k, old_v = old_layer[0], old_layer[1]
            new_k, new_v = new_layer[0], new_layer[1]
            old_len = old_k.shape[-2]

            if old_len >= new_len:
                base_k = old_k[:, :, :new_len, :].clone()
                base_v = old_v[:, :, :new_len, :].clone()
            else:
                base_k = torch.cat([old_k, new_k[:, :, old_len:new_len, :]],
                                   dim=-2).clone()
                base_v = torch.cat([old_v, new_v[:, :, old_len:new_len, :]],
                                   dim=-2).clone()

            if selected_indices.numel() > 0:
                base_k[:, :, selected_indices, :] = new_k[:, :, selected_indices, :]
                base_v[:, :, selected_indices, :] = new_v[:, :, selected_indices, :]

            blended.append((base_k, base_v, *new_layer[2:]))
        return tuple(blended)

    def _append_range_from_source_past_key_values(self, past_key_values: Any,
                                                  source_past_key_values: Any,
                                                  start_idx: int,
                                                  end_idx: int) -> Any:
        """将 source_past 的 [start_idx:end_idx) 追加到当前 past 末尾。"""
        if source_past_key_values is None or start_idx >= end_idx:
            return past_key_values

        if past_key_values is None:
            appended = []
            for src_layer in source_past_key_values:
                src_k, src_v = src_layer[0], src_layer[1]
                append_k = src_k[:, :, start_idx:end_idx, :].clone()
                append_v = src_v[:, :, start_idx:end_idx, :].clone()
                appended.append((append_k, append_v, *src_layer[2:]))
            return tuple(appended)

        merged = []
        for dst_layer, src_layer in zip(past_key_values, source_past_key_values):
            dst_k, dst_v = dst_layer[0], dst_layer[1]
            src_k, src_v = src_layer[0], src_layer[1]
            append_k = src_k[:, :, start_idx:end_idx, :]
            append_v = src_v[:, :, start_idx:end_idx, :]
            merged_k = torch.cat([dst_k, append_k], dim=-2)
            merged_v = torch.cat([dst_v, append_v], dim=-2)
            merged.append((merged_k, merged_v, *dst_layer[2:]))
        return tuple(merged)

    def _build_blended_from_old_and_selected(self, old_past_key_values: Any,
                                             selected_past_key_values: Any,
                                             selected_indices: torch.Tensor,
                                             new_len: int) -> Any:
        """近似实验：用 old KV 作底座，将 packed-selected KV scatter 回原位置。"""
        selected_num = int(selected_indices.numel())
        if selected_num <= 0:
            return self.model_runner.truncate_past_key_values(old_past_key_values,
                                                              new_len)

        blended = []
        for old_layer, sel_layer in zip(old_past_key_values, selected_past_key_values):
            old_k, old_v = old_layer[0], old_layer[1]
            sel_k, sel_v = sel_layer[0], sel_layer[1]
            old_len = old_k.shape[-2]

            if old_len >= new_len:
                base_k = old_k[:, :, :new_len, :].clone()
                base_v = old_v[:, :, :new_len, :].clone()
            else:
                pad_len = new_len - old_len
                k_pad = torch.zeros(
                    (old_k.shape[0], old_k.shape[1], pad_len, old_k.shape[3]),
                    dtype=old_k.dtype,
                    device=old_k.device,
                )
                v_pad = torch.zeros(
                    (old_v.shape[0], old_v.shape[1], pad_len, old_v.shape[3]),
                    dtype=old_v.dtype,
                    device=old_v.device,
                )
                base_k = torch.cat([old_k, k_pad], dim=-2)
                base_v = torch.cat([old_v, v_pad], dim=-2)

            # packed prefill 的序列顺序与 selected_indices 对齐，直接 scatter 回去。
            base_k[:, :, selected_indices, :] = sel_k[:, :, :selected_num, :]
            base_v[:, :, selected_indices, :] = sel_v[:, :, :selected_num, :]
            blended.append((base_k, base_v, *old_layer[2:]))
        return tuple(blended)

    def _prefill_with_kv_diff_recompute(self, req: GenerateRequest, old_tokens: List[int],
                                        old_past_key_values: Any,
                                        prompt_token_ids: List[int]) -> Tuple[torch.Tensor, Any, int]:
        """高 KV 偏差 token 重算流程（无 vLLM/PagedAttention 的纯 Transformers 实现）。"""
        # 第 1 步：对新 prompt 做一次完整 prefill，拿到 new KV（用于计算偏差分数）。
        # 说明：该实现可运行，但由于需 full pass 获取 new KV，加速收益有限。
        _, new_past_key_values = self.model_runner.forward_tokens(prompt_token_ids,
                                                                  past_key_values=None)

        old_len = self.model_runner.get_past_len(old_past_key_values)
        new_len = self.model_runner.get_past_len(new_past_key_values)
        overlap_len = min(old_len, new_len, len(old_tokens), len(prompt_token_ids))

        if overlap_len <= 0:
            logits, past_key_values = self.model_runner.forward_tokens(prompt_token_ids,
                                                                       past_key_values=None)
            return logits, past_key_values, 0

        scores = self._compute_kv_diff_scores(old_past_key_values,
                                              new_past_key_values,
                                              overlap_len=overlap_len)
        selected_indices = self._select_token_indices_from_scores(
            old_tokens=old_tokens,
            new_tokens=prompt_token_ids,
            scores=scores,
            recomp_ratio=req.recomp_ratio,
            suffix_len=req.suffix_len,
            force_changed=True,
        )

        blended_past_key_values = self._build_blended_past_key_values(
            old_past_key_values=old_past_key_values,
            new_past_key_values=new_past_key_values,
            new_len=len(prompt_token_ids),
            selected_indices=selected_indices,
        )

        # 第 2 步：基于 blended KV 重新计算“最后一个 prompt token”的 logits，进入 decode。
        if len(prompt_token_ids) == 1:
            logits, past_key_values = self.model_runner.forward_tokens(prompt_token_ids,
                                                                       past_key_values=None)
        else:
            prefix_past = self.model_runner.truncate_past_key_values(
                blended_past_key_values, len(prompt_token_ids) - 1)
            logits, past_key_values = self.model_runner.forward_tokens(
                [prompt_token_ids[-1]], past_key_values=prefix_past)

        return logits, past_key_values, int(selected_indices.numel())

    def _prefill_with_query_aware_recompute(
            self, req: GenerateRequest, old_tokens: List[int],
            old_past_key_values: Any,
            prompt_token_ids: List[int]) -> Tuple[torch.Tensor, Any, int]:
        """query-aware 近似实验：selected token 打包 prefill 后 scatter 回原位置。"""
        old_len = self.model_runner.get_past_len(old_past_key_values)
        new_len = len(prompt_token_ids)
        overlap_len = min(old_len, new_len, len(old_tokens))

        if overlap_len <= 0:
            logits, past_key_values = self.model_runner.forward_tokens(prompt_token_ids,
                                                                       past_key_values=None)
            return logits, past_key_values, 0

        query_text = req.query_text.strip()
        if query_text:
            query_token_ids = self.model_runner.encode_no_special(query_text)
        else:
            tail_start = max(0, len(prompt_token_ids) - max(req.suffix_len, 1))
            query_token_ids = prompt_token_ids[tail_start:]

        scores = self._compute_query_aware_scores(
            prompt_token_ids=prompt_token_ids,
            query_token_ids=query_token_ids,
            overlap_len=overlap_len,
        )
        qaw_variant = (req.qaw_variant or "default").lower()
        random_topk = (qaw_variant == "random_topk")
        suffix_len = 0 if qaw_variant == "no_suffix" else req.suffix_len
        random_seed = self._derive_stable_seed(req.session_id, req.qaw_random_seed)
        selected_indices = self._select_token_indices_from_scores(
            old_tokens=old_tokens,
            new_tokens=prompt_token_ids,
            scores=scores,
            recomp_ratio=req.recomp_ratio,
            suffix_len=suffix_len,
            force_changed=False,
            random_topk=random_topk,
            random_seed=random_seed,
        )
        selected_list = [int(i) for i in selected_indices.tolist()]
        selected_token_ids = [prompt_token_ids[i] for i in selected_list]
        selected_position_ids = selected_list

        # Step 1/2: 选中 token 打包一次 prefill（可选用原始 position_ids）。
        _, selected_past_key_values = self.model_runner.forward_tokens(
            selected_token_ids,
            past_key_values=None,
            position_ids=selected_position_ids,
        )

        # Step 3: 将 packed KV scatter 回完整序列位置，未选中位置复用 old KV。
        blended_past_key_values = self._build_blended_from_old_and_selected(
            old_past_key_values=old_past_key_values,
            selected_past_key_values=selected_past_key_values,
            selected_indices=selected_indices,
            new_len=new_len,
        )

        # 用 blended KV 补算最后一个 prompt token 的 logits，进入 decode。
        if new_len == 1:
            logits, past_key_values = self.model_runner.forward_tokens(
                [prompt_token_ids[0]], past_key_values=None)
        else:
            prefix_past = self.model_runner.truncate_past_key_values(
                blended_past_key_values, new_len - 1)
            logits, past_key_values = self.model_runner.forward_tokens(
                [prompt_token_ids[-1]],
                past_key_values=prefix_past,
                position_ids=[new_len - 1],
            )

        self.logger.info(
            "query-aware 打包重算: variant=%s overlap=%d selected=%d packed_len=%d",
            qaw_variant,
            overlap_len,
            int(selected_indices.numel()),
            len(selected_token_ids),
        )
        return logits, past_key_values, int(selected_indices.numel())

    @staticmethod
    def _resolve_recompute_strategy(req: GenerateRequest) -> str:
        strategy = (req.recompute_strategy or "none").lower()
        if strategy in ("none", "kv_diff", "query_aware"):
            return strategy
        # 向后兼容旧字段。
        if req.enable_kv_diff_recompute:
            return "kv_diff"
        return "none"

    def _select_recompute_indices(self, strategy: str,
                                  cand_states: torch.Tensor,
                                  query_states: torch.Tensor) -> torch.Tensor:
        """早期预留接口：当前 query-aware 主路径不再使用该函数。

        实际选择逻辑见 `_select_token_indices_from_scores`。
        """
        del strategy, query_states
        return torch.arange(cand_states.shape[0], device=cand_states.device)

    def generate(self, req: GenerateRequest) -> GenerateResult:
        t_start = time.perf_counter()
        first_token_latency_s = None
        recompute_strategy = self._resolve_recompute_strategy(req)

        prompt_token_ids = self.model_runner.encode(req.prompt)
        reused_prefix_tokens = 0
        recompute_mode = "full_prefill"
        recomputed_tokens = 0

        logits: Any = None
        past_key_values: Any = None
        prefill_token_ids: List[int] = prompt_token_ids
        cached_entry = None

        if req.use_cache:
            cached_entry = self.kv_cache.get(req.session_id)
            if cached_entry is not None and self._is_prefix(cached_entry.token_ids,
                                                            prompt_token_ids):
                reused_prefix_tokens = len(cached_entry.token_ids)
                prefill_token_ids = prompt_token_ids[reused_prefix_tokens:]
                past_key_values = cached_entry.past_key_values
                self.logger.info(
                    "命中 KV 缓存 session_id=%s, 复用前缀 token=%d, 增量 prefill token=%d",
                    req.session_id,
                    reused_prefix_tokens,
                    len(prefill_token_ids),
                )
                recompute_mode = "cache_prefix_reuse"
            elif cached_entry is not None:
                if recompute_strategy == "kv_diff":
                    self.logger.info(
                        "session_id=%s 前缀不匹配，启用 KV 偏差重算策略（ratio=%.3f, suffix_len=%d）",
                        req.session_id, req.recomp_ratio, req.suffix_len)
                    try:
                        logits, past_key_values, recomputed_tokens = self._prefill_with_kv_diff_recompute(
                            req=req,
                            old_tokens=cached_entry.token_ids,
                            old_past_key_values=cached_entry.past_key_values,
                            prompt_token_ids=prompt_token_ids,
                        )
                        prefill_token_ids = []
                        recompute_mode = "kv_diff_recompute"
                    except torch.OutOfMemoryError:
                        # kv_diff 需要额外 full pass 获取 new KV，长序列下可能峰值过高。
                        # 兜底回退到 full prefill，保证请求不中断。
                        self.logger.warning(
                            "session_id=%s KV 偏差重算触发 OOM，回退 full prefill",
                            req.session_id)
                        recompute_mode = "kv_diff_oom_fallback_full_prefill"
                        recomputed_tokens = 0
                        past_key_values = None
                        prefill_token_ids = prompt_token_ids
                        self.kv_cache.clear(req.session_id)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                elif recompute_strategy == "query_aware":
                    self.logger.info(
                        "session_id=%s 前缀不匹配，启用 Query-aware 重算策略（ratio=%.3f, suffix_len=%d）",
                        req.session_id, req.recomp_ratio, req.suffix_len)
                    logits, past_key_values, recomputed_tokens = self._prefill_with_query_aware_recompute(
                        req=req,
                        old_tokens=cached_entry.token_ids,
                        old_past_key_values=cached_entry.past_key_values,
                        prompt_token_ids=prompt_token_ids,
                    )
                    prefill_token_ids = []
                    recompute_mode = "query_aware_recompute"
                else:
                    self.logger.info("session_id=%s 前缀不匹配，回退全量 prefill",
                                     req.session_id)

        # 注意：若 prefill_token_ids 为空（新请求与缓存 prompt 完全一致），
        # MVP 为保证正确性，回退到全量 prefill。
        if len(prefill_token_ids) == 0 and past_key_values is None:
            self.logger.info("session_id=%s 新请求与缓存完全一致，回退全量 prefill",
                             req.session_id)
            reused_prefix_tokens = 0
            prefill_token_ids = prompt_token_ids
            past_key_values = None

        if len(prefill_token_ids) > 0:
            logits, past_key_values = self.model_runner.forward_tokens(
                prefill_token_ids, past_key_values=past_key_values)
        elif logits is None:
            # 命中“完整前缀复用”且缓存中携带 next-token logits 时，直接进入 decode（0 prompt 计算）。
            if (cached_entry is not None
                    and reused_prefix_tokens == len(prompt_token_ids)
                    and cached_entry.next_token_logits is not None):
                logits = cached_entry.next_token_logits
                self.logger.info("session_id=%s 复用缓存 logits，直接进入 decode",
                                 req.session_id)
            else:
                # 未携带缓存 logits 时，补算最后一个 prompt token 的 logits。
                if len(prompt_token_ids) == 1:
                    logits, past_key_values = self.model_runner.forward_tokens(
                        [prompt_token_ids[0]], past_key_values=None)
                else:
                    prefix_past = self.model_runner.truncate_past_key_values(
                        past_key_values, len(prompt_token_ids) - 1)
                    logits, past_key_values = self.model_runner.forward_tokens(
                        [prompt_token_ids[-1]], past_key_values=prefix_past)

        generated_ids: List[int] = []
        eos_id = self.model_runner.eos_token_id()

        for _ in range(req.max_new_tokens):
            next_token_id = self._sample_next_token(logits, req.temperature,
                                                    req.top_p)
            generated_ids.append(next_token_id)

            if first_token_latency_s is None:
                first_token_latency_s = time.perf_counter() - t_start

            if eos_id is not None and next_token_id == eos_id:
                self.logger.info("命中 EOS，提前结束生成")
                break

            logits, past_key_values = self.model_runner.forward_tokens(
                [next_token_id], past_key_values=past_key_values)

        # 将“本次完整上下文 + 新生成”写回会话 KV。
        if req.use_cache:
            final_token_ids = prompt_token_ids + generated_ids
            # 仅缓存“下一 token 分布”对应的最后一步 logits，避免保存整段 [T, vocab]。
            # 这对 warm/prefill（max_new_tokens=0）场景尤其重要，可显著降低显存占用。
            next_token_logits = None
            if len(generated_ids) == 0:
                next_token_logits = logits[:, -1:, :].detach().clone()
            self.kv_cache.put(
                req.session_id,
                final_token_ids,
                past_key_values,
                next_token_logits=next_token_logits,
            )

        generated_text = self.model_runner.decode(generated_ids)
        full_text = self.model_runner.decode(prompt_token_ids + generated_ids)
        total_latency_s = time.perf_counter() - t_start

        return GenerateResult(
            generated_text=generated_text,
            full_text=full_text,
            reused_prefix_tokens=reused_prefix_tokens,
            prompt_tokens=len(prompt_token_ids),
            generated_tokens=len(generated_ids),
            total_latency_s=total_latency_s,
            first_token_latency_s=first_token_latency_s,
            recompute_mode=recompute_mode,
            recomputed_tokens=recomputed_tokens,
        )
