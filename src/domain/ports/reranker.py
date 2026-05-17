"""Reranker port — contract for cross-encoder reranking adapters."""

from __future__ import annotations

from typing import Protocol

from src.domain.ports.dto import RerankRequest, RetrievedChunk


class RerankerPort(Protocol):
    """Protocol for cross-encoder reranking adapters."""

    async def rerank(self, request: RerankRequest) -> list[RetrievedChunk]:
        """Rerank retrieved chunks by cross-encoder relevance and return top-k results."""
        ...
