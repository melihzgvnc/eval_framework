"""Caching layer for evaluation results.

Provides persistent and in-memory caches keyed on the inputs to
expensive operations (claim decomposition, NLI prediction, LLM Judge
evaluation). All caches share a common interface and namespace
convention.

Typical usage::

    from eval_framework.cache import SQLiteCache, make_key

    cache = SQLiteCache()  # defaults to ~/.cache/eval_framework/cache.db

    key = make_key("judge", model="gpt-4o", sample=sample, answer=answer)
    result = cache.get_or_compute(key, lambda: judge.evaluate(sample, answer))

Backends:
- ``InMemoryCache``: dict-backed, lost on process exit. Tests and short runs.
- ``SQLiteCache``: persistent single-file cache. Default for real workloads.
"""

from .base import BaseCache, CacheStats
from .keys import make_key, parse_namespace
from .memory import InMemoryCache
from .sqlite import SQLiteCache, default_cache_path

__all__ = [
    "BaseCache",
    "CacheStats",
    "InMemoryCache",
    "SQLiteCache",
    "make_key",
    "parse_namespace",
    "default_cache_path",
]
