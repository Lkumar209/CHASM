"""API backend (Anthropic / OpenAI) — optional, behind a flag."""

from __future__ import annotations

import os
from typing import Any

from chasm.models.base import BackendConfig, GenerationResult, ModelBackend
from chasm.models.cache import ResponseCache

# Approximate cost per 1M tokens (input/output) in USD — update as needed
_COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.6),
}


class APIBackend(ModelBackend):
    """Calls a remote API; falls back gracefully if keys are absent."""

    def __init__(self, config: BackendConfig, cache: ResponseCache | None = None) -> None:
        super().__init__(config)
        self._cache = cache or ResponseCache()

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        params: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            **kwargs,
        }
        cached = self._cache.get(self.config.model_name, prompt, params)
        if cached is not None:
            return GenerationResult(text=cached, cached=True)

        if self.config.api_provider == "anthropic":
            return self._call_anthropic(prompt, params)
        elif self.config.api_provider == "openai":
            return self._call_openai(prompt, params)
        else:
            raise ValueError(f"Unknown API provider: {self.config.api_provider}")

    def _call_anthropic(self, prompt: str, params: dict[str, Any]) -> GenerationResult:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Install with: uv sync --extra api") from e

        key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=self.config.model_name,
            max_tokens=params["max_new_tokens"],
            temperature=params.get("temperature", 0.0),
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        pt, ct = msg.usage.input_tokens, msg.usage.output_tokens
        cost = self._compute_cost(pt, ct)
        self._cache.set(self.config.model_name, prompt, params, text)
        result = GenerationResult(text=text, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost)
        self._record(result)
        return result

    def _call_openai(self, prompt: str, params: dict[str, Any]) -> GenerationResult:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install with: uv sync --extra api") from e

        key = os.environ.get("OPENAI_API_KEY", "")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=self.config.model_name,
            max_tokens=params["max_new_tokens"],
            temperature=params.get("temperature", 0.0),
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        pt = resp.usage.prompt_tokens if resp.usage else 0
        ct = resp.usage.completion_tokens if resp.usage else 0
        cost = self._compute_cost(pt, ct)
        self._cache.set(self.config.model_name, prompt, params, text)
        result = GenerationResult(text=text, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost)
        self._record(result)
        return result

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_rate, out_rate = _COST_TABLE.get(self.config.model_name, (0.0, 0.0))
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000
