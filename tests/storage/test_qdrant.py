import pytest
from rag.storage.qdrant import QdrantStore
from rag.models import Chunk


@pytest.fixture
def store():
    s = QdrantStore(collection_name="test_documents")
    s.ensure_collection(dense_dim=1024)
    yield s
    s.delete_collection()


def test_ensure_collection(store):
    info = store.get_collection_info()
    assert info is not None


def test_upsert_and_search(store):
    chunk = Chunk(
        document_id="doc-1",
        content="Berlin is the capital of Germany.",
        chunk_index=0,
        token_count=7,
        metadata={"platform": "web", "author": "test"},
    )
    dense = [0.1] * 1024
    sparse_indices = [0, 5, 10]
    sparse_values = [0.8, 0.5, 0.3]

    store.upsert(
        chunk=chunk,
        dense_vector=dense,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
    )

    results = store.search(
        dense_vector=dense,
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].chunk_id == chunk.id


def test_search_with_filter(store):
    chunk = Chunk(
        document_id="doc-1",
        content="Test content",
        chunk_index=0,
        token_count=3,
        metadata={"platform": "youtube", "author": "alice"},
    )
    dense = [0.2] * 1024
    store.upsert(chunk=chunk, dense_vector=dense)

    results = store.search(
        dense_vector=dense,
        filter_platform="youtube",
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].metadata["platform"] == "youtube"
