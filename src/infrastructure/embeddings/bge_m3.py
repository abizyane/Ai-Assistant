"""BGE-M3 embedding adapter — BAAI/bge-m3 via FlagEmbedding BGEM3FlagModel."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import structlog

from src.config.settings import Settings
from src.shared.metrics import observe_histogram
from src.shared.tracing import traced

__all__ = ["BGEM3Embedder"]

log = structlog.get_logger(__name__)

_DIMENSION = 1024


class BGEM3Embedder:
    """BGE-M3 multilingual embedding adapter implementing EmbedderPort.

    Loads BGEM3FlagModel lazily on first embed call using thread-safe
    double-checked locking. Inference runs CPU-only (use_fp16=False) inside
    asyncio.to_thread to avoid blocking the event loop. Produces 1024-dim
    dense vectors.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return _DIMENSION

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

            model_id = self._settings.embedding.model
            cache_dir = self._settings.embedding.cache_dir
            self._model = BGEM3FlagModel(
                model_id,
                use_fp16=False,
                devices=["cpu"],
                cache_dir=cache_dir,
            )
            log.info("bge_m3_embedder.model_loaded", model=model_id)

    def _sync_embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model_loaded()
        batch_size = self._settings.embedding.batch_size
        with self._inference_lock:
            raw: Any = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        dense: list[list[float]] = raw["dense_vecs"].tolist()
        return dense

    @traced("embedder.embed_texts")
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model_id = self._settings.embedding.model
        batch_size = self._settings.embedding.batch_size
        start = time.perf_counter()

        result = await asyncio.to_thread(self._sync_embed, texts)

        duration = time.perf_counter() - start
        observe_histogram(
            "embedding_request_duration_seconds",
            duration,
            {"model": model_id, "batch_size": str(batch_size)},
        )

        log.info(
            "bge_m3_embedder.embed_complete",
            count=len(texts),
            duration_s=round(duration, 4),
        )

        return result

    @traced("embedder.embed_query")
    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]
