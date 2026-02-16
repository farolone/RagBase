import pytest
from rag.ingestion.web import WebIngestor
from rag.models import Platform


@pytest.fixture
def ingestor():
    return WebIngestor()


@pytest.mark.network
def test_web_ingest_live(ingestor):
    doc, chunks = ingestor.ingest("https://en.wikipedia.org/wiki/Berlin")
    assert doc.platform == Platform.WEB
    assert doc.title is not None
    assert len(chunks) >= 1
    full_text = " ".join(c.content for c in chunks)
    assert "Berlin" in full_text


@pytest.mark.network
def test_web_metadata(ingestor):
    doc, chunks = ingestor.ingest("https://en.wikipedia.org/wiki/Python_(programming_language)")
    assert doc.source_url is not None
    assert doc.metadata.get("domain") is not None
    for c in chunks:
        assert c.document_id == doc.id
