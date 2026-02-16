import pytest
from rag.retrieval.reranker import Reranker
from rag.storage.qdrant import SearchResult


def test_reranker_init():
    r = Reranker()
    assert r.base_url is not None


def test_rerank_empty():
    r = Reranker()
    results = r.rerank("test query", [])
    assert results == []


def test_rerank_no_model_returns_original_order():
    """Without a reranker model, results are returned as-is (truncated to top_k)."""
    r = Reranker(model="")
    mock_results = [
        SearchResult(chunk_id=f"c{i}", document_id="d1", content=f"content {i}", score=float(i), metadata={})
        for i in range(5)
    ]
    reranked = r.rerank("test", mock_results, top_k=3)
    assert len(reranked) == 3


@pytest.mark.network
def test_rerank_with_model():
    """Integration test - requires running LLM server with reranker model."""
    r = Reranker()
    results = [
        SearchResult(chunk_id="c1", document_id="d1", content="Berlin is the capital of Germany.", score=0.8, metadata={}),
        SearchResult(chunk_id="c2", document_id="d2", content="Pizza is a popular Italian food.", score=0.7, metadata={}),
    ]
    reranked = r.rerank("What is the capital of Germany?", results, top_k=2)
    assert len(reranked) == 2
