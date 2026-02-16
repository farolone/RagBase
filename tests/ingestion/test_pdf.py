import pytest
from pathlib import Path
from rag.ingestion.pdf import PDFIngestor
from rag.models import Platform


FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.pdf"


@pytest.fixture
def ingestor():
    return PDFIngestor()


def test_pdf_ingest_returns_document_and_chunks(ingestor):
    doc, chunks = ingestor.ingest(str(FIXTURE_PATH))
    assert doc.platform == Platform.PDF
    assert doc.title is not None
    assert len(chunks) >= 1


def test_pdf_ingest_content_extracted(ingestor):
    doc, chunks = ingestor.ingest(str(FIXTURE_PATH))
    full_text = " ".join(c.content for c in chunks)
    assert "Berlin" in full_text or "test" in full_text.lower()


def test_pdf_ingest_metadata(ingestor):
    doc, chunks = ingestor.ingest(str(FIXTURE_PATH))
    assert doc.source_url is not None
    for c in chunks:
        assert c.document_id == doc.id
