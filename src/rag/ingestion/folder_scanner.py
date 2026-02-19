"""Recursive folder scanner for document ingestion."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = [
    ".pdf", ".docx", ".pptx", ".epub", ".txt", ".md", ".html", ".xlsx", ".csv",
]

DEFAULT_EXCLUDES = [
    "__pycache__", ".git", "node_modules", ".DS_Store", "__MACOSX",
    ".venv", ".env", ".tox",
]


@dataclass
class ScannedFile:
    path: Path
    relative_path: str
    size_bytes: int
    extension: str


@dataclass
class ScanResult:
    files: list[ScannedFile] = field(default_factory=list)
    root_path: Path = field(default_factory=Path)
    skipped_count: int = 0


class FolderScanner:
    """Recursively scan folders for ingestable documents."""

    def __init__(
        self,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size_mb: float = 100,
    ):
        self.extensions = set(ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                              for ext in (extensions or DEFAULT_EXTENSIONS))
        self.exclude_patterns = set(exclude_patterns or DEFAULT_EXCLUDES)
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

    def scan(self, root: str | Path) -> ScanResult:
        """Recursively scan a directory for matching files."""
        root_path = Path(root).resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {root_path}")

        result = ScanResult(root_path=root_path)

        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue

            # Check exclude patterns against any part of the path
            if any(excl in path.parts for excl in self.exclude_patterns):
                result.skipped_count += 1
                continue

            ext = path.suffix.lower()
            if ext not in self.extensions:
                result.skipped_count += 1
                continue

            size = path.stat().st_size
            if size > self.max_file_size_bytes:
                logger.info(f"Skipping {path} ({size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.0f}MB)")
                result.skipped_count += 1
                continue

            if size == 0:
                result.skipped_count += 1
                continue

            result.files.append(ScannedFile(
                path=path,
                relative_path=str(path.relative_to(root_path)),
                size_bytes=size,
                extension=ext,
            ))

        logger.info(f"Scanned {root_path}: {len(result.files)} files, {result.skipped_count} skipped")
        return result
