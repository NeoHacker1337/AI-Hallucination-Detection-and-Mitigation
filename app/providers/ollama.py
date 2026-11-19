"""Ollama backend (local models such as llama3, mistral, phi3)."""

from __future__ import annotations

import time

import httpx

from app.config import get_settings
from app.providers.base import Completion, DecodingParams, LLMProvider, Message, ProviderError


class OllamaProvider(LLMProvider):
    """Calls the Ollama `/api/chat` endpoint without streaming."""

    prefix = "ollama"

    def __init__(self, model: str, base_url: str | None = None) -> None:
        super().__init__(model)
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = settings.llm_timeout_s
        self.max_retries = settings.llm_max_retries

    def check_available(self) -> None:
        """Verify the Ollama server is reachable and the model has been pulled."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.base_url} ({exc}). Start it with `ollama serve` "
                "or set OLLAMA_BASE_URL."
            ) from exc
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            raise ProviderError(
                f"Ollama model '{self.model}' is not installed. Run `ollama pull {self.model}`. "
                f"Installed models: {sorted(names) or 'none'}"
            )

    def chat(self, messages: list[Message], params: DecodingParams) -> Completion:
        """Send a chat request and return the completion with token counts."""
        options: dict = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "num_predict": params.max_tokens,
        }
        if params.seed is not None:
            options["seed"] = params.seed
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": options,
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
                if resp.status_code == 404:
                    raise ProviderError(
                        f"Ollama returned 404 for model '{self.model}': {resp.text[:200]}. "
                        f"Run `ollama pull {self.model}`."
                    )
                resp.raise_for_status()
                data = resp.json()
                latency = (time.perf_counter() - start) * 1000.0
                return Completion(
                    text=data.get("message", {}).get("content", ""),
                    model=self.spec,
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                    latency_ms=latency,
                    raw={k: v for k, v in data.items() if k != "message"},
                )
            except ProviderError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 10))
        raise ProviderError(f"Ollama call failed after {self.max_retries + 1} attempts: {last_exc}")
