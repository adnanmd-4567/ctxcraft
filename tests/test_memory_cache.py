import json
import os
import pytest
from ctxcraft.memory_cache import MemoryCache, MemoryCacheError

@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / "test_memory.json")

def test_set_and_get(cache_path):
    cache = MemoryCache(path=cache_path)
    cache.set("goal", "book a flight to Tokyo")
    assert cache.get("goal") == "book a flight to Tokyo"

def test_get_missing_key_returns_default(cache_path):
    cache = MemoryCache(path=cache_path)
    assert cache.get("nonexistent") is None
    assert cache.get("nonexistent", default="fallback") == "fallback"

def test_delete_removes_key(cache_path):
    cache = MemoryCache(path=cache_path)
    cache.set("temp_note", "delete me")
    cache.delete("temp_note")
    assert cache.get("temp_note") is None
    assert "temp_note" not in cache

def test_persistence_across_simulated_restart(cache_path):
    cache1 = MemoryCache(path=cache_path)
    cache1.set("step", 3)
    cache1.set("last_tool_result", {"status": "ok", "data": [1, 2, 3]})

    cache2 = MemoryCache(path=cache_path)
    assert cache2.get("step") == 3
    assert cache2.get("last_tool_result") == {"status": "ok", "data": [1, 2, 3]}

def test_auto_flush_false_requires_explicit_flush(cache_path):
    cache1 = MemoryCache(path=cache_path, auto_flush=False)
    cache1.set("draft_note", "not yet saved")

    cache2 = MemoryCache(path=cache_path)
    assert cache2.get("draft_note") is None

    cache1.flush()
    cache3 = MemoryCache(path=cache_path)
    assert cache3.get("draft_note") == "not yet saved"

def test_clear_empties_store_and_persists(cache_path):
    cache = MemoryCache(path=cache_path)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()

    cache_reloaded = MemoryCache(path=cache_path)
    assert len(cache_reloaded) == 0

def test_keys_returns_all_stored_keys(cache_path):
    cache = MemoryCache(path=cache_path)
    cache.set("x", 1)
    cache.set("y", 2)
    assert set(cache.keys()) == {"x", "y"}

def test_corrupted_file_raises_clear_error(cache_path):
    with open(cache_path, "w") as f:
        f.write("{not valid json")

    with pytest.raises(MemoryCacheError):
        MemoryCache(path=cache_path)

def test_len_reflects_store_size(cache_path):
    cache = MemoryCache(path=cache_path)
    assert len(cache) == 0
    cache.set("a", 1)
    cache.set("b", 2)
    assert len(cache) == 2