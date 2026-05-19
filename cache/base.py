"""Base cache abstraction.

Defines the interface all cache backends must implement and provides
a ``CacheStats`` container for hit/miss tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheStats:
    """Hit/miss statistics for a cache instance."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    by_namespace: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that hit the cache (0.0–1.0)."""
        if self.total_lookups == 0:
            return 0.0
        return self.hits / self.total_lookups

    def record_hit(self, namespace: str) -> None:
        self.hits += 1
        self._bump(namespace, "hits")

    def record_miss(self, namespace: str) -> None:
        self.misses += 1
        self._bump(namespace, "misses")

    def record_set(self, namespace: str) -> None:
        self.sets += 1
        self._bump(namespace, "sets")

    def _bump(self, namespace: str, field_name: str) -> None:
        bucket = self.by_namespace.setdefault(
            namespace, {"hits": 0, "misses": 0, "sets": 0}
        )
        bucket[field_name] = bucket.get(field_name, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "total_lookups": self.total_lookups,
            "hit_rate": self.hit_rate,
            "by_namespace": dict(self.by_namespace),
        }


class BaseCache(ABC):
    """Abstract base for all cache backends.

    Implementations must be safe to use from a single process. Backends
    that need cross-process safety (e.g., SQLite) should document that
    explicitly.
    """

    def __init__(self) -> None:
        self.stats = CacheStats()

    # ------------------------------------------------------------------
    # Backend-specific operations (must be implemented)
    # ------------------------------------------------------------------

    @abstractmethod
    def _get(self, key: str) -> Optional[Any]:
        """Return the raw stored value for key, or None if absent."""

    @abstractmethod
    def _set(self, key: str, value: Any) -> None:
        """Persist value under key, overwriting any existing entry."""

    @abstractmethod
    def _delete(self, key: str) -> bool:
        """Remove key. Return True if it existed."""

    @abstractmethod
    def _has(self, key: str) -> bool:
        """Return True if key exists in the backend."""

    @abstractmethod
    def _clear(self, namespace: Optional[str] = None) -> int:
        """Remove all entries (optionally filtered by namespace).

        Returns the number of entries removed.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Total number of entries in the cache."""

    # ------------------------------------------------------------------
    # Public API (track stats, delegate to backend)
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Look up a key. Records hit/miss stats."""
        from eval_framework.cache.keys import parse_namespace

        ns = parse_namespace(key)
        value = self._get(key)
        if value is None:
            self.stats.record_miss(ns)
        else:
            self.stats.record_hit(ns)
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value."""
        from eval_framework.cache.keys import parse_namespace

        self._set(key, value)
        self.stats.record_set(parse_namespace(key))

    def has(self, key: str) -> bool:
        """Check existence without recording hit/miss."""
        return self._has(key)

    def delete(self, key: str) -> bool:
        """Remove a single entry. Returns True if it existed."""
        return self._delete(key)

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear all entries (optionally restricted to a namespace).

        Args:
            namespace: If given, only entries with this prefix are removed.

        Returns:
            Number of entries removed.
        """
        return self._clear(namespace)

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
    ) -> T:
        """Atomic get-or-compute helper.

        If the key exists, returns the cached value. Otherwise calls
        ``compute_fn``, stores its result, and returns it.

        Args:
            key: Cache key (typically from ``make_key``).
            compute_fn: Zero-arg callable that produces the value on miss.

        Returns:
            Cached or freshly computed value.

        Raises:
            Whatever ``compute_fn`` raises. Failures are NOT cached.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        value = compute_fn()
        # Don't cache None — None is the sentinel for "not in cache".
        if value is not None:
            self.set(key, value)
        return value

    # Context manager support for backends that hold resources
    def __enter__(self) -> "BaseCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release any resources. Default is no-op."""
