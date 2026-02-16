import httpx

from rag.config import settings


class OllamaClient:
    """Generic Ollama REST API client supporting multiple models."""

    def __init__(self, host: str | None = None):
        self.host = host or f"http://{settings.ollama_host}:{settings.ollama_port}"

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ) -> str:
        model = model or settings.ollama_model_rag

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        response = httpx.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
        timeout: float = 120.0,
    ) -> dict:
        model = model or settings.ollama_model_rag

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        response = httpx.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def is_available(self, model: str | None = None) -> bool:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            if response.status_code != 200:
                return False
            if model:
                models = [m["name"] for m in response.json().get("models", [])]
                return any(model in m for m in models)
            return True
        except Exception:
            return False
