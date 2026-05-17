from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import structlog

from src.config.settings import Settings
from src.domain.ports.dto import RerankRequest, RetrievedChunk
from src.shared.metrics import observe_histogram, set_gauge
from src.shared.tracing import traced

__all__ = ["BGEReranker"]

log = structlog.get_logger(__name__)


class BGEReranker:
    """BGE-Reranker-v2-m3 cross-encoder reranker adapter implementing RerankerPort.

    Loads the FlagReranker model lazily (on first rerank call) using thread-safe
    double-checked locking. All inference runs CPU-only (use_fp16=False) inside
    asyncio.to_thread to avoid blocking the event loop.
    """

    def __init__(self, settings: Settings) -> None:
        """Store settings; model is NOT loaded at construction time.

        Args:
            settings: Application settings providing reranker configuration.
        """
        self._settings = settings
        self._model: Any = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _ensure_model_loaded(self) -> None:
        """Load FlagReranker thread-safely via double-checked locking.

        Imports FlagEmbedding and creates the model instance on the very first
        call. Subsequent calls return immediately without acquiring the lock.
        """
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from FlagEmbedding import FlagReranker

            model_id = self._settings.reranker.model
            self._model = FlagReranker(model_id, use_fp16=False)
            log.info("bge_reranker.model_loaded", model=model_id)

    @traced("reranker.rerank")
    async def rerank(self, request: RerankRequest) -> list[RetrievedChunk]:
        """Rerank candidate chunks by cross-encoder relevance and return top-k.

        Delegates synchronous FlagReranker.compute_score to a thread-pool
        worker via asyncio.to_thread so the event loop is never blocked.

        Args:
            request: Query, candidate chunks, and desired top_k cutoff.

        Returns:
            Chunks sorted by cross-encoder score descending, limited to top_k.
        """
        start = time.perf_counter()
        set_gauge("reranker_candidates_count", float(len(request.chunks)))

        result = await asyncio.to_thread(self._sync_rerank, request)

        duration = time.perf_counter() - start
        observe_histogram("reranker_request_duration_seconds", duration)

        log.info(
            "bge_reranker.rerank_complete",
            query_length=len(request.query),
            candidates=len(request.chunks),
            top_k=request.top_k,
            returned=len(result),
            duration_s=round(duration, 4),
        )

        return result

    def _sync_rerank(self, request: RerankRequest) -> list[RetrievedChunk]:
        """Execute synchronous cross-encoder scoring and sort results.

        Runs inside a thread-pool worker (via asyncio.to_thread). Handles the
        edge case where compute_score returns a bare float for a single pair.

        Args:
            request: Query, candidate chunks, and desired top_k cutoff.

        Returns:
            Top-k chunks with updated scores, sorted by relevance descending.
        """
        if not request.chunks:
            return []

        self._ensure_model_loaded()

        pairs = [[request.query, chunk.content] for chunk in request.chunks]
        with self._inference_lock:
            raw: list[float] | float = self._model.compute_score(
                pairs,
                batch_size=self._settings.reranker.batch_size,
                normalize=True,
            )
        scores: list[float] = [raw] if isinstance(raw, float) else raw

        scored = sorted(
            zip(scores, request.chunks, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )

        threshold = self._settings.reranker.min_score
        above = [(s, c) for s, c in scored if s >= threshold]
        # If nothing clears the relevance bar, return an empty list so the
        # generate node can emit a graceful "not found" reply without an LLM call.
        top = above[: request.top_k]
        return [chunk.model_copy(update={"score": score}) for score, chunk in top]
