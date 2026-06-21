"""Echo backend for CI / smoke tests — returns deterministic fake completions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chasm.models.base import BackendConfig, GenerationResult, ModelBackend

if TYPE_CHECKING:
    from chasm.models.cache import ResponseCache


class EchoBackend(ModelBackend):
    """Returns the first 80 chars of the prompt as completion. For tests only."""

    def __init__(self, config: BackendConfig, cache: ResponseCache | None = None) -> None:
        super().__init__(config)

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        text = f"[ECHO] {prompt[:80]}"
        return GenerationResult(text=text, prompt_tokens=len(prompt.split()), completion_tokens=5)
