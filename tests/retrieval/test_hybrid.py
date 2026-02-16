import pytest
from rag.retrieval.hybrid import HybridRetriever
from rag.processing.embedding import Embedder
from rag.storage.qdrant import QdrantStore
from rag.models import Chunk


@pytest.fixture(scope="module")
def retriever():
    embedder = Embedder()
    store = QdrantStore(collection_name="test_hybrid")
    store.ensure_collection(dense_dim=1024)

    # Index some test documents
    texts = [
        ("Berlin is the capital of Germany and has 3.7 million inhabitants.", {"platform": "web"}),
        ("Machine learning models can process natural language text.", {"platform": "pdf"}),
        ("Die Hauptstadt von Deutschland ist Berlin.", {"platform": "web"}),
    ]
    for text, meta in texts:
        emb = embedder.embed(text)
        chunk = Chunk(
            document_id="doc-test",
            content=text,
            chunk_index=0,
            token_count=len(text.split()),
            metadata=meta,
        )
        store.upsert(
            chunk=chunk,
            dense_vector=emb.dense,
            sparse_indices=emb.sparse_indices,
            sparse_values=emb.sparse_values,
        )

    r = HybridRetriever(embedder=embedder, store=store)
    yield r
    store.delete_collection()


def test_retrieve_relevant(retriever):
    results = retriever.retrieve("What is the capital of Germany?", limit=5)
    assert len(results) >= 1
    # Should return Berlin-related content
    top_content = results[0].content.lower()
    assert "berlin" in top_content or "germany" in top_content or "deutschland" in top_content


def test_retrieve_with_filter(retriever):
    results = retriever.retrieve(
        "machine learning",
        filter_platform="pdf",
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].metadata["platform"] == "pdf"
