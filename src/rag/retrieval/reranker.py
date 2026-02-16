import httpx
import time

from rag.config import settings
from rag.storage.qdrant import SearchResult


class Reranker:
    """Reranks search results using LLM-based scoring via OpenAI-compatible API."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.llm_model_reranker
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not results:
            return []

        # Skip reranking if no reranker model configured
        if not self.model:
            return results[:top_k]

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
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 10,
                    "stream": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
            for token in text.split():
                try:
                    return float(token)
                except ValueError:
                    continue
            return 0.0
        except Exception:
            return 0.0
