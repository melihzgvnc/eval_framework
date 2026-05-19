"""In-memory cache backend.

Useful for tests and short-lived processes. Not persistent across runs.
Thread-safe via a simple lock.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from eval_framework.cache.base import BaseCache


class InMemoryCache(BaseCache):
    """Dict-backed cache. Lost on process exit."""

    def __init__(self) -> None:
        super().__init__()
        self._store: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def _get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._store.get(key)

    def _set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def _delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def _has(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def _clear(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is None:
                count = len(self._store)
                self._store.clear()
                return count

            prefix = f"{namespace}:"
            keys_to_delete = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._store[k]
            return len(keys_to_delete)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
