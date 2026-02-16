from pathlib import Path

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform
from rag.processing.chunking import HierarchicalChunker


class PDFIngestor(BaseIngestor):
    def __init__(self):
        self.chunker = HierarchicalChunker()

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        path = Path(source)
        text = self._extract_text(path)

        doc = Document(
            title=path.stem,
            source_url=str(path.resolve()),
            platform=Platform.PDF,
            metadata={"filename": path.name},
        )

        chunks = self.chunker.chunk(
            text=text,
            document_id=doc.id,
            metadata={"platform": "pdf", "source": str(path)},
        )
        return doc, chunks

    def _extract_text(self, path: Path) -> str:
        # Try Docling first for high-quality extraction
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(path))
            return result.document.export_to_markdown()
        except Exception:
            pass

        # Fallback to PyMuPDF4LLM
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(path))
        except Exception:
            pass

        # Last resort: basic pymupdf
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from {path}: {e}")
