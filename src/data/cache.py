import hashlib
import json
import os
import sqlite3
import time
from typing import Any


class Cache:
    def __init__(self, db_path: str, default_ttl_seconds: int = 86400):
        self.db_path = db_path
        self.default_ttl_seconds = default_ttl_seconds
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    query TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON cache(source)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _make_key(source: str, query: str, params: dict) -> str:
        params_str = json.dumps(params, sort_keys=True)
        raw = f"{source}:{query}:{params_str}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _params_hash(params: dict) -> str:
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    def set(
        self,
        source: str,
        query: str,
        params: dict,
        data: Any,
        pinned: bool = False,
        ttl_seconds: int | None = None,
    ):
        key = self._make_key(source, query, params)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache
                   (cache_key, source, query, params_hash, data, created_at, ttl_seconds, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (key, source, query, self._params_hash(params),
                 json.dumps(data), time.time(), ttl, int(pinned)),
            )

    def get(self, source: str, query: str, params: dict) -> Any | None:
        key = self._make_key(source, query, params)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data, created_at, ttl_seconds, pinned FROM cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        data_str, created_at, ttl_seconds, pinned = row
        if not pinned and (time.time() - created_at) > ttl_seconds:
            self._delete_key(key)
            return None
        return json.loads(data_str)

    def unpin(self, source: str, query: str, params: dict):
        key = self._make_key(source, query, params)
        with self._connect() as conn:
            conn.execute("UPDATE cache SET pinned = 0 WHERE cache_key = ?", (key,))

    def clear_source(self, source: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE source = ?", (source,))

    def clear_all(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM cache")

    def size_bytes(self) -> int:
        if not os.path.exists(self.db_path):
            return 0
        return os.path.getsize(self.db_path)

    def _delete_key(self, key: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
