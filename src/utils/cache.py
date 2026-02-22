"""Lightweight SQLite key-value cache for phrase resolutions."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from src.utils.config import DEFAULT_CACHE_PATH


class Cache:
    """Simple persistent cache backed by SQLite."""

    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created_at REAL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        """Return cached value or None."""
        row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        """Store value under key (overwrites existing)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO cache(key, value, created_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Remove all cached entries."""
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
