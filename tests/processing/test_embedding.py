import pytest
from rag.processing.embedding import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embed_single(embedder):
    result = embedder.embed("Hello World")
    assert result.dense is not None
    assert len(result.dense) == 1024
    assert result.sparse_indices is not None
    assert result.sparse_values is not None


def test_embed_batch(embedder):
    texts = ["Hello", "World", "Berlin ist die Hauptstadt"]
    results = embedder.embed_batch(texts)
    assert len(results) == 3
    for r in results:
        assert len(r.dense) == 1024


def test_embed_german_english_similarity(embedder):
    de = embedder.embed("Hund")
    en = embedder.embed("dog")
    import numpy as np
    similarity = np.dot(de.dense, en.dense) / (
        np.linalg.norm(de.dense) * np.linalg.norm(en.dense)
    )
    assert similarity > 0.5  # Cross-lingual similarity should be meaningful
