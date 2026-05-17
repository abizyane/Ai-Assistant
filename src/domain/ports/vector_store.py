"""Vector store port — contract for hybrid search storage adapters."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from src.domain.ports.dto import ChunkWithEmbedding, RetrievedChunk


class VectorStorePort(Protocol):
    """Protocol for hybrid vector store adapters."""

    async def upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        """Upsert chunks into the vector store and return the count of affected rows."""
        ...

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform dense ANN search and return ranked chunks."""
        ...

    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform a hybrid dense+sparse search and return ranked chunks."""
        ...

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks belonging to the given document and return the count removed."""
        ...
