"""Content hashing for trajectory deduplication and determinism checks."""

from __future__ import annotations

import json
from typing import Any

import xxhash


def content_hash(obj: Any) -> str:
    """Return a stable hex digest of any JSON-serialisable object."""
    serialised = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return xxhash.xxh64(serialised.encode()).hexdigest()
