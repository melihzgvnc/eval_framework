"""SQLite cache backend.

Persistent, single-file cache using stdlib ``sqlite3``. Safe for
concurrent reads and serialized writes within a single process; safe
across processes thanks to SQLite's locking.

Default location: ``~/.cache/eval_framework/cache.db`` (XDG-friendly).
Override via the ``path`` constructor argument.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional, Union

from eval_framework.cache.base import BaseCache


def default_cache_path() -> Path:
    """Return the default SQLite cache file path.

    Honors ``XDG_CACHE_HOME`` if set, otherwise ``~/.cache``.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "eval_framework" / "cache.db"


class SQLiteCache(BaseCache):
    """SQLite-backed persistent cache.

    Stores values as JSON text. Use this for any non-trivial run where
    you want results to survive process restarts.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_cache_namespace ON cache(namespace);
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        super().__init__()
        self.path = Path(path) if path else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Single connection, serialized via lock. SQLite's own locking
        # handles cross-process safety; the lock prevents intra-process
        # races on the connection object.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we use explicit transactions
        )
        # WAL improves concurrent reader/writer behavior.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(self._SCHEMA)

    # ------------------------------------------------------------------
    # Backend implementation
    # ------------------------------------------------------------------

    def _get(self, key: str) -> Optional[Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def _set(self, key: str, value: Any) -> None:
        from eval_framework.cache.keys import parse_namespace

        ns = parse_namespace(key)
        payload = json.dumps(value, default=str, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, namespace, value) "
                "VALUES (?, ?, ?)",
                (key, ns, payload),
            )

    def _delete(self, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM cache WHERE key = ?", (key,)
            )
            return cur.rowcount > 0

    def _has(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    def _clear(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace is None:
                cur = self._conn.execute("DELETE FROM cache")
            else:
                cur = self._conn.execute(
                    "DELETE FROM cache WHERE namespace = ?", (namespace,)
                )
            return cur.rowcount

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM cache"
            ).fetchone()
        return row[0]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.ProgrammingError:
                # Already closed
                pass

    # ------------------------------------------------------------------
    # Backend-specific helpers
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        """Reclaim disk space after large deletes."""
        with self._lock:
            self._conn.execute("VACUUM")

    def namespace_counts(self) -> dict:
        """Number of entries per namespace, useful for debugging."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT namespace, COUNT(*) FROM cache GROUP BY namespace"
            ).fetchall()
        return {ns: count for ns, count in rows}
