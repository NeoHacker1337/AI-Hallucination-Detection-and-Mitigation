"""Provider interface shared by every LLM backend."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """Raised when an LLM backend is unreachable, misconfigured or returns an error."""


@dataclass
class Message:
    """One chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Completion:
    """Result of one LLM call, with the accounting needed for cost metrics."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        """Prompt + completion tokens (unknown counts are treated as 0)."""
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@dataclass
class DecodingParams:
    """Decoding settings. Kept identical across configurations to ensure a fair comparison."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int | None = 42


class LLMProvider(abc.ABC):
    """Abstract LLM backend.

    Implementations must be thread-safe for `chat` because experiments may call it from worker
    threads.
    """

    #: Provider prefix used in model specs, e.g. "ollama" in "ollama:llama3".
    prefix: str = ""

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def spec(self) -> str:
        """Full model spec, e.g. "ollama:llama3"."""
        return f"{self.prefix}:{self.model}"

    @abc.abstractmethod
    def chat(self, messages: list[Message], params: DecodingParams) -> Completion:
        """Run one chat completion."""

    @abc.abstractmethod
    def check_available(self) -> None:
        """Raise ProviderError with a clear message if the model cannot be used."""

    def complete(self, prompt: str, params: DecodingParams, system: str | None = None) -> Completion:
        """Convenience wrapper for a single-turn prompt."""
        messages: list[Message] = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.chat(messages, params)
