import re

from rag.config import settings
from rag.generation.llm import OllamaClient


class QueryRouter:
    """Routes queries to appropriate model based on complexity.

    Simple queries -> Qwen 2.5-72B (low hallucination, fast)
    Complex/multi-step -> MiniMax M2.5 orchestrates, Qwen synthesizes
    Code questions -> MiniMax M2.5
    Long context (>128K) -> MiniMax M2.5 (200K context window)
    """

    COMPLEX_PATTERNS = [
        r"compare",
        r"vergleich",
        r"analyse",
        r"analyze",
        r"step.by.step",
        r"schritt.f.r.schritt",
        r"multiple.sources",
        r"across.all",
        r"zusammenfass",
        r"summarize.all",
    ]

    CODE_PATTERNS = [
        r"code",
        r"function",
        r"implement",
        r"debug",
        r"error",
        r"programming",
    ]

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self.model_rag = settings.ollama_model_rag
        self.model_agent = settings.ollama_model_agent

    def route(self, query: str, context_length: int = 0) -> str:
        query_lower = query.lower()

        # Long context -> MiniMax (200K window)
        if context_length > 128_000:
            return self.model_agent

        # Code questions -> MiniMax
        if any(re.search(p, query_lower) for p in self.CODE_PATTERNS):
            return self.model_agent

        # Complex multi-step queries -> MiniMax
        if any(re.search(p, query_lower) for p in self.COMPLEX_PATTERNS):
            return self.model_agent

        # Default -> Qwen (reliable, low hallucination)
        return self.model_rag

    def generate(
        self,
        query: str,
        context: str = "",
        system: str | None = None,
    ) -> str:
        context_length = len(context)
        model = self.route(query, context_length)

        prompt = query
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}"

        return self.client.generate(
            prompt=prompt,
            model=model,
            system=system,
        )
