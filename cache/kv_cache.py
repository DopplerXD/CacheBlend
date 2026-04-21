"""KV 缓存管理器。

同时保留旧的 session 级缓存接口，并提供新的 chunk 级 KV 池。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time
from typing import Any, Dict, List, Optional


@dataclass
class SessionKV:
    """单个会话缓存单元。"""

    token_ids: List[int]
    past_key_values: Any
    next_token_logits: Any
    last_access_ts: float


@dataclass
class ChunkKV:
    """单个文本块缓存单元。"""

    cache_key: str
    namespace: str
    token_ids: List[int]
    past_key_values: Any
    last_access_ts: float


class KVCacheManager:
    """内存 KV 缓存管理器。

    设计目标：
    1. 极简实现，优先可读性。
    2. 仅支持单进程内存缓存。
    3. 使用 LRU + TTL，避免缓存无限增长。
    """

    def __init__(self,
                 max_sessions: int,
                 ttl_seconds: int,
                 logger,
                 max_chunks: Optional[int] = None) -> None:
        self.max_sessions = max_sessions
        self.max_chunks = max_chunks if max_chunks is not None else max_sessions * 128
        self.ttl_seconds = ttl_seconds
        self.logger = logger
        self._store: "OrderedDict[str, SessionKV]" = OrderedDict()
        self._chunk_store: "OrderedDict[str, ChunkKV]" = OrderedDict()

    def _is_expired(self, entry: SessionKV) -> bool:
        return (time.time() - entry.last_access_ts) > self.ttl_seconds

    def _touch(self, session_id: str, entry: SessionKV) -> None:
        entry.last_access_ts = time.time()
        self._store[session_id] = entry
        self._store.move_to_end(session_id)

    def _evict_expired(self) -> None:
        expired_keys = [
            sid for sid, entry in self._store.items() if self._is_expired(entry)
        ]
        for sid in expired_keys:
            self.logger.info("KV 过期淘汰 session_id=%s", sid)
            self._store.pop(sid, None)

    def _evict_lru_if_needed(self) -> None:
        while len(self._store) > self.max_sessions:
            old_sid, _ = self._store.popitem(last=False)
            self.logger.info("KV LRU 淘汰 session_id=%s", old_sid)
        while len(self._chunk_store) > self.max_chunks:
            old_key, old_entry = self._chunk_store.popitem(last=False)
            self.logger.info("chunk KV LRU 淘汰 key=%s namespace=%s",
                             old_key, old_entry.namespace)

    def get(self, session_id: str) -> Optional[SessionKV]:
        self._evict_expired()
        entry = self._store.get(session_id)
        if entry is None:
            return None
        self._touch(session_id, entry)
        return entry

    def put(self,
            session_id: str,
            token_ids: List[int],
            past_key_values: Any,
            next_token_logits: Any = None) -> None:
        entry = SessionKV(token_ids=token_ids,
                          past_key_values=past_key_values,
                          next_token_logits=next_token_logits,
                          last_access_ts=time.time())
        self._store[session_id] = entry
        self._store.move_to_end(session_id)
        self._evict_lru_if_needed()

    @staticmethod
    def build_chunk_key(namespace: str, model_name: str,
                        token_ids: List[int]) -> str:
        """为 chunk token 序列生成稳定缓存 key。"""
        payload = ",".join(str(t) for t in token_ids)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        model_digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        return f"{namespace}:{model_digest}:{digest}"

    def _evict_expired_chunks(self) -> None:
        expired_keys = [
            key for key, entry in self._chunk_store.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            entry = self._chunk_store.pop(key, None)
            namespace = entry.namespace if entry is not None else ""
            self.logger.info("chunk KV 过期淘汰 key=%s namespace=%s", key,
                             namespace)

    def get_chunk(self, cache_key: str) -> Optional[ChunkKV]:
        self._evict_expired_chunks()
        entry = self._chunk_store.get(cache_key)
        if entry is None:
            return None
        entry.last_access_ts = time.time()
        self._chunk_store[cache_key] = entry
        self._chunk_store.move_to_end(cache_key)
        return entry

    def put_chunk(self, cache_key: str, namespace: str, token_ids: List[int],
                  past_key_values: Any) -> None:
        entry = ChunkKV(
            cache_key=cache_key,
            namespace=namespace,
            token_ids=list(token_ids),
            past_key_values=past_key_values,
            last_access_ts=time.time(),
        )
        self._chunk_store[cache_key] = entry
        self._chunk_store.move_to_end(cache_key)
        self._evict_lru_if_needed()

    def clear(self, session_id: str) -> None:
        if session_id in self._store:
            self._store.pop(session_id, None)
            self.logger.info("KV 主动清理 session_id=%s", session_id)

    def clear_chunks(self, namespace: Optional[str] = None) -> None:
        if namespace is None:
            self._chunk_store.clear()
            self.logger.info("chunk KV 全量清理")
            return
        keys = [
            key for key, entry in self._chunk_store.items()
            if entry.namespace == namespace
        ]
        for key in keys:
            self._chunk_store.pop(key, None)
        self.logger.info("chunk KV 清理 namespace=%s count=%d", namespace,
                         len(keys))

    def size(self) -> int:
        self._evict_expired()
        return len(self._store)

    def stats(self) -> Dict[str, int]:
        self._evict_expired()
        self._evict_expired_chunks()
        return {
            "sessions": len(self._store),
            "chunks": len(self._chunk_store),
            "max_sessions": self.max_sessions,
            "max_chunks": self.max_chunks,
            "ttl_seconds": self.ttl_seconds,
        }
