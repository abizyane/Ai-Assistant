from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langdetect import LangDetectException, detect  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.domain.ports.chunker import ChunkerPort
from src.domain.ports.doc_loader import DocLoaderPort
from src.domain.ports.dto import ChunkWithEmbedding, IngestionReport
from src.domain.ports.embedder import EmbedderPort
from src.domain.ports.tracer import TracerPort
from src.domain.ports.vector_store import VectorStorePort
from src.infrastructure.persistence.models import DocumentORM, IngestionRunORM
from src.shared.metrics import inc_counter, observe_histogram
from src.shared.tracing import traced

__all__ = ["IngestDocumentsUseCase"]

_HASH_CHUNK_BYTES = 8192
_UNKNOWN_LANG = "unknown"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_language(text: str) -> str:
    try:
        return str(detect(text))
    except LangDetectException:
        return _UNKNOWN_LANG


class IngestDocumentsUseCase:
    """Orchestrates the document ingestion pipeline.

    Loads source files from *path*, detects language per chunk, embeds in
    batches, upserts into the vector store and records an
    :class:`~src.infrastructure.persistence.models.IngestionRunORM` row for
    each file processed.  Files whose SHA-256 hash already exists in
    ``ingestion_runs`` are skipped (idempotency).

    Args:
        loader: Adapter that loads raw documents from a file path.
        chunker: Adapter that splits a raw document into text chunks.
        embedder: Adapter that embeds a batch of texts into dense vectors.
        vector_store: Adapter that stores chunks with their embeddings.
        session_repo: SQLAlchemy async session factory used for persistence.
        tracer: Distributed tracing adapter.
        logger: Standard-library logger instance.
    """

    def __init__(
        self,
        loader: DocLoaderPort,
        chunker: ChunkerPort,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
        session_repo: async_sessionmaker[AsyncSession],
        tracer: TracerPort,
        logger: logging.Logger,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._session_factory = session_repo
        self._tracer = tracer
        self._logger = logger

    @traced("ingest_documents.execute")
    async def execute(
        self,
        path: Path,
        *,
        language_hint: str | None = None,
    ) -> IngestionReport:
        """Run the ingestion pipeline for all PDF files under *path*.

        Steps per file:
        1. Compute SHA-256; skip if already in ``ingestion_runs``.
        2. Load raw documents via the loader adapter.
        3. Chunk each document; detect language per chunk (or use *language_hint*).
        4. Embed all chunks in one batch call.
        5. Upsert :class:`ChunkWithEmbedding` objects into the vector store.
        6. Record an :class:`IngestionRunORM` row with status / chunk count.

        Per-file errors are caught and appended to the report; processing
        continues for remaining files.

        Args:
            path: Directory or single file to ingest.
            language_hint: Override language detection for every chunk when set.

        Returns:
            :class:`IngestionReport` with counts and any error tuples.
        """
        start = time.perf_counter()
        files_processed = 0
        files_skipped = 0
        chunks_created = 0
        errors: list[tuple[str, str]] = []

        source_files = self._collect_files(path)

        for file_path in source_files:
            try:
                file_hash = _hash_file(file_path)

                if await self._is_already_ingested(file_hash):
                    files_skipped += 1
                    continue

                file_chunks = await self._process_file(file_path, file_hash, language_hint)
                chunks_created += file_chunks
                files_processed += 1

                inc_counter("ingestion_files_total")

            except Exception as exc:
                self._logger.error("Ingestion failed for %s: %s", file_path, exc)
                errors.append((str(file_path), str(exc)))

        inc_counter("ingestion_chunks_total", amount=chunks_created)

        duration = time.perf_counter() - start
        observe_histogram("ingestion_duration_seconds", duration)

        return IngestionReport(
            files_processed=files_processed,
            files_skipped=files_skipped,
            chunks_created=chunks_created,
            duration_seconds=duration,
            errors=errors,
        )

    def _collect_files(self, path: Path) -> list[Path]:
        """Return all supported source files under *path* (PDF, Markdown, HTML)."""
        supported_suffixes = (".pdf", ".md", ".html")
        if path.is_dir():
            collected: list[Path] = []
            for suffix in supported_suffixes:
                collected.extend(path.rglob(f"*{suffix}"))
            return sorted(collected)
        if path.is_file() and path.suffix.lower() in supported_suffixes:
            return [path]
        return []

    async def _is_already_ingested(self, file_hash: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                select(IngestionRunORM).where(IngestionRunORM.source_hash == file_hash)
            )
            return result.scalar_one_or_none() is not None

    async def _process_file(
        self,
        file_path: Path,
        file_hash: str,
        language_hint: str | None,
    ) -> int:
        """Load, chunk, embed and upsert a single file; record ingestion run.

        Args:
            file_path: Absolute path to the source file.
            file_hash: Pre-computed SHA-256 hex digest.
            language_hint: Optional language override for every chunk.

        Returns:
            Total number of chunks created across all documents in the file.
        """
        documents = await self._loader.load(str(file_path))
        total_chunks = 0

        for doc in documents:
            chunks = await self._chunker.chunk([doc])  # type: ignore[misc, arg-type]
            if not chunks:
                continue

            langs = [language_hint or _detect_language(c.content) for c in chunks]
            texts = [c.content for c in chunks]
            embeddings = await self._embedder.embed_texts(texts)

            doc_id = uuid.uuid5(uuid.NAMESPACE_URL, doc.content_hash)

            doc_language = language_hint or langs[0] if langs else "en"
            await self._upsert_document(
                doc_id=doc_id,
                source_path=str(file_path),
                content_hash=doc.content_hash,
                language=doc_language,
            )

            chunks_with_emb: list[ChunkWithEmbedding] = [
                ChunkWithEmbedding(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.content_hash}-{chunk.position}"),
                    document_id=doc_id,
                    content=chunk.content,
                    embedding=emb,
                    position=chunk.position,
                    token_count=chunk.token_count,
                    source_path=str(file_path),
                    metadata={**chunk.metadata, "language": lang},
                )
                for chunk, emb, lang in zip(chunks, embeddings, langs, strict=True)
            ]

            await self._vector_store.upsert(chunks_with_emb)
            total_chunks += len(chunks)

        await self._record_ingestion_run(file_path, file_hash, total_chunks)
        return total_chunks

    async def _record_ingestion_run(
        self,
        file_path: Path,
        file_hash: str,
        chunks_created: int,
    ) -> None:
        run = IngestionRunORM(
            source_path=str(file_path),
            source_hash=file_hash,
            status="completed",
            chunks_created=chunks_created,
            finished_at=datetime.now(UTC),
        )
        async with self._session_factory() as session:
            session.add(run)
            await session.commit()

    async def _upsert_document(
        self,
        *,
        doc_id: uuid.UUID,
        source_path: str,
        content_hash: str,
        language: str,
    ) -> None:
        """Ensure a parent :class:`DocumentORM` row exists for *doc_id*.

        Chunks reference documents via a foreign key, so the parent row must
        be persisted before any chunk upsert. Re-ingestion of the same content
        hash is a no-op.
        """
        async with self._session_factory() as session:
            existing = await session.execute(select(DocumentORM).where(DocumentORM.id == doc_id))
            if existing.scalar_one_or_none() is not None:
                return
            session.add(
                DocumentORM(
                    id=doc_id,
                    source_path=source_path,
                    content_hash=content_hash,
                    language=language,
                )
            )
            await session.commit()
