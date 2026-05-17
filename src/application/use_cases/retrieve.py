"""Retrieve use case — embed query, hybrid search, and optional reranking."""

from __future__ import annotations

import structlog
from langdetect import LangDetectException, detect  # type: ignore[import-untyped]

from src.config.settings import Settings
from src.domain.ports.dto import RerankRequest, RetrievedChunk
from src.domain.ports.embedder import EmbedderPort
from src.domain.ports.reranker import RerankerPort
from src.domain.ports.vector_store import VectorStorePort
from src.shared.tracing import traced

__all__ = ["RetrieveUseCase"]

_UNKNOWN_LANG = "unknown"

log = structlog.get_logger(__name__)


def _detect_language(text: str) -> str:
    """Return ISO-639-1 language code for *text*, or 'unknown' on failure."""
    try:
        return str(detect(text))
    except LangDetectException:
        return _UNKNOWN_LANG


class RetrieveUseCase:
    """Orchestrates query embedding, hybrid vector search, and optional reranking.

    Args:
        embedder: Adapter that embeds query text into dense vectors.
        vector_store: Adapter that performs hybrid dense+sparse retrieval.
        reranker: Adapter that reranks retrieved chunks via cross-encoder.
        settings: Application settings for top-k configuration.
    """

    def __init__(
        self,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        reranker: RerankerPort,
        settings: Settings,
    ) -> None:
        """Store injected dependencies."""
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._settings = settings

    @traced("use_case.retrieve")
    async def execute(
        self,
        query: str,
        language: str | None = None,
        session_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Execute retrieval pipeline: detect language, embed, search, rerank.

        Steps:
        1. Detect language via langdetect if ``language`` is None.
        2. Embed query via the embedder adapter.
        3. Perform hybrid search using the vector store.
        4. Rerank results if count exceeds ``top_k_rerank``.

        Args:
            query: Natural-language question to answer.
            language: BCP-47 language code; auto-detected when None.
            session_id: Optional session identifier for tracing context.

        Returns:
            List of retrieved (and optionally reranked) chunks.
        """
        lang = language if language is not None else _detect_language(query)
        log.debug("retrieve.language_detected", language=lang, query_len=len(query))

        query_vector = await self._embedder.embed_query(query)

        vs = self._settings.vector_store
        chunks = await self._vector_store.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            top_k=vs.top_k_dense,
            filters={"language": lang},
        )

        log.debug("retrieve.hybrid_search_done", chunks_found=len(chunks))

        if not chunks:
            return chunks

        if len(chunks) > vs.top_k_rerank:
            chunks = await self._reranker.rerank(
                RerankRequest(query=query, chunks=chunks, top_k=vs.top_k_rerank)
            )
            log.debug("retrieve.reranked", chunks_kept=len(chunks))

        return chunks
