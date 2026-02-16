import re

from rag.storage.qdrant import SearchResult


class CitationGenerator:
    """Builds prompts with citations and parses citation references."""

    SYSTEM_PROMPT = (
        "You are a helpful research assistant. Answer the question based on the provided sources. "
        "Always cite your sources using [1], [2], etc. inline. "
        "If you cannot answer from the provided sources, say so."
    )

    def build_prompt(self, query: str, results: list[SearchResult]) -> tuple[str, dict[int, SearchResult]]:
        citation_map = {}
        context_parts = []

        for i, result in enumerate(results, 1):
            citation_map[i] = result
            source_info = result.metadata.get("source_url", result.metadata.get("platform", "unknown"))
            context_parts.append(f"[{i}] {result.content}\n(Source: {source_info})")

        context = "\n\n".join(context_parts)
        prompt = f"Sources:\n{context}\n\nQuestion: {query}"

        return prompt, citation_map

    def parse_citations(self, answer: str, citation_map: dict[int, SearchResult]) -> dict:
        cited_refs = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))

        sources = []
        for ref in sorted(cited_refs):
            if ref in citation_map:
                result = citation_map[ref]
                sources.append({
                    "ref": ref,
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "content_preview": result.content[:200],
                    "source_url": result.metadata.get("source_url", ""),
                })

        return {
            "answer": answer,
            "sources": sources,
            "citation_count": len(sources),
        }
