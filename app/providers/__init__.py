"""LLM providers behind one interface. Model specs look like "ollama:llama3" or "openai:gpt-4o-mini"."""

from __future__ import annotations

import threading

from app.providers.base import Completion, DecodingParams, LLMProvider, Message, ProviderError
from app.providers.ollama import OllamaProvider
from app.providers.openai_compat import OpenAICompatProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    OllamaProvider.prefix: OllamaProvider,
    OpenAICompatProvider.prefix: OpenAICompatProvider,
}
_CACHE: dict[str, LLMProvider] = {}
_LOCK = threading.Lock()


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split "provider:model" into its parts. A bare name defaults to Ollama.

    Model names may themselves contain ':' (e.g. "ollama:llama3:8b"), so only the first ':' splits.
    """
    spec = spec.strip()
    if not spec:
        raise ProviderError("Empty model spec.")
    prefix, sep, rest = spec.partition(":")
    if sep and prefix in _REGISTRY:
        return prefix, rest
    return OllamaProvider.prefix, spec


def get_provider(spec: str) -> LLMProvider:
    """Return a (cached) provider instance for a model spec."""
    prefix, model = parse_model_spec(spec)
    key = f"{prefix}:{model}"
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = _REGISTRY[prefix](model)
        return _CACHE[key]


__all__ = [
    "Completion",
    "DecodingParams",
    "LLMProvider",
    "Message",
    "ProviderError",
    "get_provider",
    "parse_model_spec",
]
