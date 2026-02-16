import re

from rag.config import settings
from rag.generation.llm import LLMClient


class QueryRouter:
    """Routes queries to appropriate model based on complexity."""

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

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()
        self.model_rag = settings.llm_model_rag
        self.model_agent = settings.llm_model_agent

    def route(self, query: str, context_length: int = 0) -> str:
        query_lower = query.lower()

        if context_length > 128_000:
            return self.model_agent

        if any(re.search(p, query_lower) for p in self.CODE_PATTERNS):
            return self.model_agent

        if any(re.search(p, query_lower) for p in self.COMPLEX_PATTERNS):
            return self.model_agent

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
