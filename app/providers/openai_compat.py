"""Backend for any OpenAI-compatible `/chat/completions` API (OpenAI, vLLM, LM Studio, Groq, ...)."""

from __future__ import annotations

import time

import httpx

from app.config import get_settings
from app.providers.base import Completion, DecodingParams, LLMProvider, Message, ProviderError


class OpenAICompatProvider(LLMProvider):
    """Calls `{OPENAI_BASE_URL}/chat/completions` with a bearer token."""

    prefix = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        super().__init__(model)
        settings = get_settings()
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.timeout = settings.llm_timeout_s
        self.max_retries = settings.llm_max_retries

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_available(self) -> None:
        """Send a one-token request to confirm the endpoint, key and model all work."""
        try:
            self.chat([Message("user", "Reply with OK.")], DecodingParams(temperature=0.0, max_tokens=2))
        except ProviderError as exc:
            raise ProviderError(
                f"OpenAI-compatible model '{self.model}' at {self.base_url} is not usable: {exc}. "
                "Check OPENAI_BASE_URL, OPENAI_API_KEY and the model name."
            ) from exc

    def chat(self, messages: list[Message], params: DecodingParams) -> Completion:
        """Send a chat completion request with retries on rate limits and server errors."""
        payload: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            payload["seed"] = params.seed
        last_err = ""
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except httpx.HTTPError as exc:
                last_err = str(exc)
                time.sleep(min(2**attempt, 20))
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            if resp.status_code >= 400:
                raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            latency = (time.perf_counter() - start) * 1000.0
            choices = data.get("choices") or []
            if not choices:
                raise ProviderError(f"Response had no choices: {str(data)[:300]}")
            usage = data.get("usage") or {}
            return Completion(
                text=(choices[0].get("message") or {}).get("content") or "",
                model=self.spec,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                latency_ms=latency,
                raw={"id": data.get("id"), "finish_reason": choices[0].get("finish_reason")},
            )
        raise ProviderError(f"OpenAI-compatible call failed after retries: {last_err}")
