"""Embedder port — contract for text embedding model adapters."""

from __future__ import annotations

from typing import Protocol


class EmbedderPort(Protocol):
    """Protocol for text embedding model adapters."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their dense vector representations."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text and return its dense vector representation."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension produced by this model."""
        ...
