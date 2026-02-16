from abc import ABC, abstractmethod
from rag.models import Document, Chunk


class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        """Ingest a source and return a Document with its Chunks."""
        ...
