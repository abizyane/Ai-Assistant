"""Document loader port — contract for source document ingestion adapters."""

from __future__ import annotations

from typing import Protocol

from src.domain.ports.dto import RawDocument


class DocLoaderPort(Protocol):
    """Protocol for document loading adapters that ingest source files."""

    async def load(self, source_path: str) -> list[RawDocument]:
        """Load documents from the given source path and return raw document objects."""
        ...

    def supports(self, source_path: str) -> bool:
        """Return True if this loader can handle the given source path or file type."""
        ...
