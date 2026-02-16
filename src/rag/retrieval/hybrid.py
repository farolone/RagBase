from rag.processing.embedding import Embedder, EmbeddingResult
from rag.storage.qdrant import QdrantStore, SearchResult


class HybridRetriever:
    """Hybrid retrieval combining dense + sparse search via Qdrant."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore()

    def retrieve(
        self,
        query: str,
        limit: int = 20,
        filter_platform: str | None = None,
        filter_author: str | None = None,
    ) -> list[SearchResult]:
        embedding = self.embedder.embed(query)

        results = self.store.search(
            dense_vector=embedding.dense,
            sparse_indices=embedding.sparse_indices,
            sparse_values=embedding.sparse_values,
            filter_platform=filter_platform,
            filter_author=filter_author,
            limit=limit,
        )

        return results
