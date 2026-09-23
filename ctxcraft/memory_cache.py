import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

class MemoryCacheError(Exception):
    pass

class MemoryCache:
    def __init__(self, path: str = ".ctxcraft_memory.json", auto_flush: bool = True):
        self.path = Path(path)
        self.auto_flush = auto_flush
        self._store: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise MemoryCacheError(f"Failed to load memory cache from {self.path}: {e}")
        else:
            self._store = {}

    def flush(self) -> None:
        try:
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)
            os.replace(tmp_path, self.path)
        except OSError as e:
            raise MemoryCacheError(f"Failed to write memory cache to {self.path}: {e}")

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        if self.auto_flush:
            self.flush()

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def delete(self, key: str) -> None:
        if key in self._store:
            del self._store[key]
            if self.auto_flush:
                self.flush()

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def clear(self) -> None:
        self._store = {}
        if self.auto_flush:
            self.flush()

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)