"""Unit tests for model backends."""

from pathlib import Path

import pytest

from chasm.models.base import BackendConfig
from chasm.models.cache import ResponseCache
from chasm.models.echo_backend import EchoBackend
from chasm.models.factory import build_backend


def test_echo_backend_returns_text() -> None:
    cfg = BackendConfig(backend="echo", model_name="echo")
    backend = EchoBackend(cfg)
    result = backend.generate("Hello world")
    assert "Hello world" in result.text


def test_echo_backend_not_cached() -> None:
    cfg = BackendConfig(backend="echo", model_name="echo")
    backend = EchoBackend(cfg)
    result = backend.generate("test prompt")
    assert not result.cached


def test_factory_builds_echo() -> None:
    cfg = BackendConfig(backend="echo")
    backend = build_backend(cfg)
    assert isinstance(backend, EchoBackend)


def test_factory_unknown_backend() -> None:
    cfg = BackendConfig(backend="nonexistent")
    with pytest.raises(ValueError, match="Unknown backend"):
        build_backend(cfg)


def test_response_cache_roundtrip(tmp_path: Path) -> None:
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    assert cache.get("m", "p", {}) is None
    cache.set("m", "p", {}, "hello")
    assert cache.get("m", "p", {}) == "hello"
    assert cache.size == 1


def test_response_cache_key_includes_params(tmp_path: Path) -> None:
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.set("m", "p", {"t": 0.5}, "A")
    cache.set("m", "p", {"t": 1.0}, "B")
    assert cache.get("m", "p", {"t": 0.5}) == "A"
    assert cache.get("m", "p", {"t": 1.0}) == "B"
    assert cache.size == 2
