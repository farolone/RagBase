from datetime import datetime

import trafilatura

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform
from rag.processing.chunking import HierarchicalChunker


class WebIngestor(BaseIngestor):
    def __init__(self):
        self.chunker = HierarchicalChunker()

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        downloaded = trafilatura.fetch_url(source)
        if downloaded is None:
            raise RuntimeError(f"Failed to fetch URL: {source}")

        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            with_metadata=True,
        )

        if result is None:
            raise RuntimeError(f"Failed to extract content from: {source}")

        # Extract metadata from trafilatura
        metadata_obj = trafilatura.extract(
            downloaded,
            output_format="xmltei",
            with_metadata=True,
        )

        meta = trafilatura.metadata.extract_metadata(downloaded)
        title = meta.title if meta and meta.title else source
        author = meta.author if meta else None
        date = meta.date if meta else None

        # Parse published_at from date string
        published_at = self._parse_date(date)

        doc = Document(
            title=title,
            source_url=source,
            platform=Platform.WEB,
            author=author,
            created_at=published_at,
            metadata={
                "domain": source.split("//")[-1].split("/")[0] if "//" in source else "",
                "date": date,
            },
        )

        chunks = self.chunker.chunk(
            text=result,
            document_id=doc.id,
            metadata={"platform": "web", "source_url": source},
        )

        return doc, chunks

    @staticmethod
    def _parse_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except (ValueError, IndexError):
            return None
