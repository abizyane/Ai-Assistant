"""multilingual-E5-small embedding adapter — intfloat/multilingual-e5-small."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, cast

import structlog

from src.config.settings import Settings
from src.shared.metrics import observe_histogram
from src.shared.tracing import traced

__all__ = ["MultilingualE5Embedder"]

log = structlog.get_logger(__name__)

_DIMENSION = 384


class MultilingualE5Embedder:
    """multilingual-E5-small embedding adapter implementing EmbedderPort.

    Loads SentenceTransformer lazily on first embed call using thread-safe
    double-checked locking. Inference runs CPU-only inside asyncio.to_thread
    to avoid blocking the event loop. Produces 384-dim dense vectors.

    Prefix rules (required by the E5 family):
    - ``embed_query`` prepends ``"query: "`` before encoding.
    - ``embed_texts`` prepends ``"passage: "`` to every text before encoding.
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
            from sentence_transformers import SentenceTransformer

            model_id = self._settings.embedding.model
            cache_dir = self._settings.embedding.cache_dir
            self._model = SentenceTransformer(model_id, cache_folder=cache_dir, device="cpu")
            log.info("multilingual_e5_embedder.model_loaded", model=model_id)

    def _sync_embed(self, prefixed_texts: list[str]) -> list[list[float]]:
        self._ensure_model_loaded()
        batch_size = self._settings.embedding.batch_size
        with self._inference_lock:
            raw: Any = self._model.encode(
                prefixed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
            )
        return cast(list[list[float]], raw.tolist())

    @traced("embedder.embed_texts")
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model_id = self._settings.embedding.model
        batch_size = self._settings.embedding.batch_size
        start = time.perf_counter()

        prefixed = ["passage: " + t for t in texts]
        result = await asyncio.to_thread(self._sync_embed, prefixed)

        duration = time.perf_counter() - start
        observe_histogram(
            "embedding_request_duration_seconds",
            duration,
            {"model": model_id, "batch_size": str(batch_size)},
        )

        log.info(
            "multilingual_e5_embedder.embed_complete",
            count=len(texts),
            duration_s=round(duration, 4),
        )

        return result

    @traced("embedder.embed_query")
    async def embed_query(self, text: str) -> list[float]:
        model_id = self._settings.embedding.model
        batch_size = self._settings.embedding.batch_size
        start = time.perf_counter()

        prefixed = ["query: " + text]
        results = await asyncio.to_thread(self._sync_embed, prefixed)

        duration = time.perf_counter() - start
        observe_histogram(
            "embedding_request_duration_seconds",
            duration,
            {"model": model_id, "batch_size": str(batch_size)},
        )

        log.info(
            "multilingual_e5_embedder.embed_complete",
            count=1,
            duration_s=round(duration, 4),
        )

        return results[0]
