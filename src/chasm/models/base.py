"""Abstract model backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendConfig:
    backend: str = "local"  # "local" | "api"
    model_name: str = "Qwen/Qwen3-4B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    load_in_4bit: bool = False
    device: str = "cpu"
    api_provider: str = "anthropic"  # used when backend="api"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class ModelBackend(ABC):
    """Common interface for all model backends."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult: ...

    def accounting(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "cost_usd": self._total_cost_usd,
        }

    def _record(self, result: GenerationResult) -> None:
        if not result.cached:
            self._total_prompt_tokens += result.prompt_tokens
            self._total_completion_tokens += result.completion_tokens
            self._total_cost_usd += result.cost_usd
