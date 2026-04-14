"""会话级 KV 缓存（简单 LRU + TTL）。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional


@dataclass
class SessionKV:
    """单个会话缓存单元。"""

    token_ids: List[int]
    past_key_values: Any
    last_access_ts: float


class KVCacheManager:
    """内存 KV 缓存管理器。

    设计目标：
    1. 极简实现，优先可读性。
    2. 仅支持单进程内存缓存。
    3. 使用 LRU + TTL，避免会话无限增长。
    """

    def __init__(self, max_sessions: int, ttl_seconds: int, logger) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.logger = logger
        self._store: "OrderedDict[str, SessionKV]" = OrderedDict()

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

    def get(self, session_id: str) -> Optional[SessionKV]:
        self._evict_expired()
        entry = self._store.get(session_id)
        if entry is None:
            return None
        self._touch(session_id, entry)
        return entry

    def put(self, session_id: str, token_ids: List[int], past_key_values: Any) -> None:
        entry = SessionKV(token_ids=token_ids,
                          past_key_values=past_key_values,
                          last_access_ts=time.time())
        self._store[session_id] = entry
        self._store.move_to_end(session_id)
        self._evict_lru_if_needed()

    def clear(self, session_id: str) -> None:
        if session_id in self._store:
            self._store.pop(session_id, None)
            self.logger.info("KV 主动清理 session_id=%s", session_id)

    def size(self) -> int:
        self._evict_expired()
        return len(self._store)

    def stats(self) -> Dict[str, int]:
        self._evict_expired()
        return {
            "sessions": len(self._store),
            "max_sessions": self.max_sessions,
            "ttl_seconds": self.ttl_seconds,
        }
