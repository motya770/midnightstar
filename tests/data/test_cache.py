import time
import tempfile
import os
from src.data.cache import Cache


def test_cache_set_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        cache.set("gwas", "SP4", {"param": "v1"}, {"result": "data"})
        result = cache.get("gwas", "SP4", {"param": "v1"})
        assert result == {"result": "data"}


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        result = cache.get("gwas", "NONEXISTENT", {})
        assert result is None


def test_cache_ttl_expiry():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path, default_ttl_seconds=1)
        cache.set("gwas", "SP4", {}, {"result": "data"})
        assert cache.get("gwas", "SP4", {}) is not None
        time.sleep(1.1)
        assert cache.get("gwas", "SP4", {}) is None


def test_cache_pin_survives_ttl():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path, default_ttl_seconds=1)
        cache.set("gwas", "SP4", {}, {"result": "data"}, pinned=True)
        time.sleep(1.1)
        assert cache.get("gwas", "SP4", {}) == {"result": "data"}


def test_cache_unpin():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path, default_ttl_seconds=1)
        cache.set("gwas", "SP4", {}, {"result": "data"}, pinned=True)
        cache.unpin("gwas", "SP4", {})
        time.sleep(1.1)
        assert cache.get("gwas", "SP4", {}) is None


def test_cache_clear_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        cache.set("gwas", "SP4", {}, {"r": 1})
        cache.set("gtex", "SP4", {}, {"r": 2})
        cache.clear_source("gwas")
        assert cache.get("gwas", "SP4", {}) is None
        assert cache.get("gtex", "SP4", {}) is not None


def test_cache_size_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cache = Cache(db_path)
        cache.set("gwas", "SP4", {}, {"r": 1})
        size = cache.size_bytes()
        assert size > 0
