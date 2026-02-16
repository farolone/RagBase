import httpx

from rag.config import settings
from rag.storage.qdrant import SearchResult


class Reranker:
    """Reranks search results using Qwen3-Reranker via Ollama API."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or settings.ollama_model_reranker
        self.host = host or f"http://{settings.ollama_host}:{settings.ollama_port}"

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return []

        scored = []
        for result in results:
            score = self._score_pair(query, result.content)
            scored.append((score, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def _score_pair(self, query: str, document: str) -> float:
        prompt = (
            f"Given the query: '{query}'\n\n"
            f"Rate the relevance of this document on a scale of 0-10:\n"
            f"'{document}'\n\n"
            f"Return ONLY a number between 0 and 10."
        )

        try:
            response = httpx.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
                timeout=30.0,
            )
            response.raise_for_status()
            text = response.json().get("response", "0").strip()
            # Extract first number from response
            for token in text.split():
                try:
                    return float(token)
                except ValueError:
                    continue
            return 0.0
        except Exception:
            return 0.0
