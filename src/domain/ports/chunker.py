"""Chunker port — contract for document chunking adapters."""

from __future__ import annotations

from typing import Protocol

from src.domain.ports.dto import ChunkContent, RawDocument


class ChunkerPort(Protocol):
    """Protocol for document chunking adapters that split raw text into chunks."""

    def chunk(self, document: RawDocument) -> list[ChunkContent]:
        """Split a raw document into a list of ordered text chunks."""
        ...
