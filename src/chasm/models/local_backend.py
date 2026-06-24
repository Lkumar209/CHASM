"""Local HuggingFace transformers backend."""

from __future__ import annotations

from typing import Any

from chasm.models.base import BackendConfig, GenerationResult, ModelBackend
from chasm.models.cache import ResponseCache


class LocalBackend(ModelBackend):
    """Runs an HF model locally; lazy-loads on first call."""

    def __init__(self, config: BackendConfig, cache: ResponseCache | None = None) -> None:
        super().__init__(config)
        self._cache = cache or ResponseCache()
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise RuntimeError(
                "Local backend requires 'torch' and 'transformers'. "
                "Install with: uv sync --extra local"
            ) from e

        quant_cfg = None
        if self.config.load_in_4bit:
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quant_cfg,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        params: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            **kwargs,
        }
        cached = self._cache.get(self.config.model_name, prompt, params)
        if cached is not None:
            return GenerationResult(text=cached, cached=True)

        self._load()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, **params)
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        self._cache.set(self.config.model_name, prompt, params, text)
        result = GenerationResult(
            text=text,
            prompt_tokens=inputs["input_ids"].shape[1],
            completion_tokens=len(new_tokens),
        )
        self._record(result)
        return result
