import time
from collections.abc import AsyncIterator

import httpx

from rag.config import settings


class LLMClient:
    """OpenAI-compatible LLM client with retry for unstable networks."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.max_retries = settings.llm_max_retries

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return result["choices"][0]["message"]["content"]

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
        timeout: float | None = None,
    ) -> dict:
        model = model or settings.llm_model_rag
        timeout = timeout or settings.llm_timeout

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        return self._request("/chat/completions", payload, timeout)

    async def stream_generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM using SSE. Yields content strings."""
        model = model or settings.llm_model_rag
        timeout = timeout or settings.llm_timeout

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                import json
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def is_available(self, model: str | None = None) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=10.0)
            if response.status_code != 200:
                return False
            if model:
                models = [m["id"] for m in response.json().get("data", [])]
                return any(model in m for m in models)
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=10.0)
            response.raise_for_status()
            return [m["id"] for m in response.json().get("data", [])]
        except Exception:
            return []

    def _request(self, endpoint: str, payload: dict, timeout: float) -> dict:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = httpx.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
            except httpx.HTTPStatusError:
                raise
        raise last_error


# Backward compatibility alias
OllamaClient = LLMClient
