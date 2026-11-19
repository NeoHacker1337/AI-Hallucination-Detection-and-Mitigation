"""Per-run tracing: stage outputs, latency, LLM calls and tokens."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.providers import Completion, DecodingParams, LLMProvider, Message


@dataclass
class StageRecord:
    """Measurements for one executed stage."""

    stage: str
    order: int
    output: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Trace:
    """Wraps a provider so every LLM call is attributed to the currently open stage."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.stages: list[StageRecord] = []
        self._current: StageRecord | None = None
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        """Open a stage; LLM calls inside it are counted toward it."""
        if self._current is not None:
            raise RuntimeError(f"Stage '{name}' opened inside stage '{self._current.stage}'.")
        record = StageRecord(stage=name, order=len(self.stages))
        self._current = record
        start = time.perf_counter()
        try:
            yield record
        finally:
            record.latency_ms = (time.perf_counter() - start) * 1000.0
            self._current = None
            self.stages.append(record)

    def chat(self, messages: list[Message], params: DecodingParams) -> Completion:
        """Run an LLM call and record its usage."""
        completion = self.provider.chat(messages, params)
        if self._current is not None:
            self._current.llm_calls += 1
            self._current.prompt_tokens += completion.prompt_tokens or 0
            self._current.completion_tokens += completion.completion_tokens or 0
        return completion

    def complete(self, prompt: str, params: DecodingParams, system: str | None = None) -> Completion:
        """Single-turn convenience wrapper around `chat`."""
        messages = [Message("system", system)] if system else []
        messages.append(Message("user", prompt))
        return self.chat(messages, params)

    @property
    def total_latency_ms(self) -> float:
        """Wall-clock time since the trace was created."""
        return (time.perf_counter() - self._started) * 1000.0

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.stages)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.stages)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.stages)
