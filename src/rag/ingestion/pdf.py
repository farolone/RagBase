"""PDF Ingestor — backward-compatible shim delegating to DocumentIngestor."""

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform


class PDFIngestor(BaseIngestor):
    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        from rag.ingestion.document import DocumentIngestor
        doc_ingestor = DocumentIngestor()
        doc, chunks = doc_ingestor.ingest(source)
        doc.platform = Platform.PDF
        return doc, chunks
