"""Build the right backend from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chasm.models.base import BackendConfig, ModelBackend
    from chasm.models.cache import ResponseCache


def build_backend(config: BackendConfig, cache: ResponseCache | None = None) -> ModelBackend:
    if config.backend == "local":
        from chasm.models.local_backend import LocalBackend
        return LocalBackend(config, cache)
    elif config.backend == "api":
        from chasm.models.api_backend import APIBackend
        return APIBackend(config, cache)
    elif config.backend == "echo":
        from chasm.models.echo_backend import EchoBackend
        return EchoBackend(config, cache)
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}. Choose 'local', 'api', or 'echo'.")
