from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag.config import settings


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict


class QdrantStore:
    def __init__(self, collection_name: str | None = None):
        self.client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self.collection_name = collection_name or settings.qdrant_collection

    def ensure_collection(self, dense_dim: int = 1024):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_dim, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            for field, schema in [
                ("platform", PayloadSchemaType.KEYWORD),
                ("author", PayloadSchemaType.KEYWORD),
                ("document_id", PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )

    def get_collection_info(self):
        return self.client.get_collection(self.collection_name)

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)

    def upsert(
        self,
        chunk,
        dense_vector: list[float],
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
    ):
        vectors = {"dense": dense_vector}
        if sparse_indices and sparse_values:
            vectors["sparse"] = SparseVector(
                indices=sparse_indices, values=sparse_values
            )

        payload = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            **chunk.metadata,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=hash(chunk.id) % (2**63),
                    vector=vectors,
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        dense_vector: list[float],
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        filter_platform: str | None = None,
        filter_author: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        conditions = []
        if filter_platform:
            conditions.append(
                FieldCondition(
                    key="platform", match=MatchValue(value=filter_platform)
                )
            )
        if filter_author:
            conditions.append(
                FieldCondition(
                    key="author", match=MatchValue(value=filter_author)
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            SearchResult(
                chunk_id=hit.payload["chunk_id"],
                document_id=hit.payload["document_id"],
                content=hit.payload["content"],
                score=hit.score,
                metadata={
                    k: v
                    for k, v in hit.payload.items()
                    if k not in ("chunk_id", "document_id", "content", "chunk_index")
                },
            )
            for hit in results.points
        ]
