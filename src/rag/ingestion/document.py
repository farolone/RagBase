"""Unified document ingestor for local files (PDF, EPUB, DOCX, PPTX, TXT, etc.)."""

import logging
from pathlib import Path

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform
from rag.processing.chunking import HierarchicalChunker

logger = logging.getLogger(__name__)

# Extensions handled by Docling
DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".xlsx", ".md", ".adoc"}

# Extensions read as plain text
PLAINTEXT_EXTENSIONS = {".txt", ".log", ".csv", ".json", ".yaml", ".yml"}

# All supported extensions
SUPPORTED_EXTENSIONS = DOCLING_EXTENSIONS | PLAINTEXT_EXTENSIONS | {".epub"}


class DocumentIngestor(BaseIngestor):
    """Unified ingestor for all local file types."""

    def __init__(self):
        self.chunker = HierarchicalChunker()
        self._docling_converter = None

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        text = self._extract_text(path)
        if not text or not text.strip():
            raise ValueError(f"No text extracted from {path}")

        # Use Platform.PDF for .pdf files (backward compat), DOCUMENT for everything else
        platform = Platform.PDF if ext == ".pdf" else Platform.DOCUMENT

        doc = Document(
            title=path.stem,
            source_url=str(path),
            platform=platform,
            metadata={
                "file_size": path.stat().st_size,
                "extension": ext,
                "folder_path": str(path.parent),
            },
        )

        chunks = self.chunker.chunk(
            text=text,
            document_id=doc.id,
            metadata={"platform": platform.value, "source": str(path)},
        )
        return doc, chunks

    def _extract_text(self, path: Path) -> str:
        """Dispatch text extraction based on file extension."""
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf(path)
        elif ext == ".epub":
            return self._extract_epub(path)
        elif ext in PLAINTEXT_EXTENSIONS:
            return self._extract_plaintext(path)
        elif ext in DOCLING_EXTENSIONS:
            return self._extract_with_docling(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf(self, path: Path) -> str:
        """PDF extraction with Docling -> PyMuPDF4LLM -> PyMuPDF fallback chain."""
        # Try Docling first for high-quality extraction
        try:
            return self._extract_with_docling(path)
        except Exception as e:
            logger.debug(f"Docling failed for {path}: {e}")

        # Fallback to PyMuPDF4LLM
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(path))
        except Exception as e:
            logger.debug(f"PyMuPDF4LLM failed for {path}: {e}")

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

    def _extract_with_docling(self, path: Path) -> str:
        """Extract text using Docling (supports PDF, DOCX, PPTX, HTML, XLSX, MD, AsciiDoc)."""
        if self._docling_converter is None:
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()

        result = self._docling_converter.convert(str(path))
        return result.document.export_to_markdown()

    def _extract_epub(self, path: Path) -> str:
        """Extract text from EPUB using ebooklib + BeautifulSoup."""
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(str(path), options={"ignore_ncx": True})
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text:
                texts.append(text)

        return "\n\n".join(texts)

    def _extract_plaintext(self, path: Path) -> str:
        """Read plain text files (UTF-8)."""
        return path.read_text(encoding="utf-8")
