"""Disk-backed response cache keyed by (model, prompt, sampling params)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xxhash


class ResponseCache:
    """Simple file-based cache; each entry is a JSON file named by key hash."""

    def __init__(self, cache_dir: str | Path = ".cache/responses") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, prompt: str, params: dict[str, Any]) -> str:
        payload = json.dumps({"model": model, "prompt": prompt, "params": params}, sort_keys=True)
        return xxhash.xxh64(payload.encode()).hexdigest()

    def get(self, model: str, prompt: str, params: dict[str, Any]) -> str | None:
        path = self._dir / f"{self._key(model, prompt, params)}.json"
        if path.exists():
            data: dict[str, str] = json.loads(path.read_text())
            return data["text"]
        return None

    def set(self, model: str, prompt: str, params: dict[str, Any], text: str) -> None:
        path = self._dir / f"{self._key(model, prompt, params)}.json"
        path.write_text(json.dumps({"model": model, "text": text}))

    @property
    def size(self) -> int:
        return sum(1 for _ in self._dir.glob("*.json"))
