"""PGVector hybrid (dense + FTS + RRF) vector store adapter."""

from __future__ import annotations

import time
from uuid import UUID

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config.settings import Settings
from src.domain.ports.dto import ChunkWithEmbedding, RetrievedChunk
from src.infrastructure.persistence.models import ChunkORM, DocumentORM
from src.shared.metrics import inc_counter, observe_histogram
from src.shared.tracing import traced

__all__ = ["PGVectorStore"]

_RRF_K: int = 60


class PGVectorStore:
    """PostgreSQL + pgvector vector store implementing VectorStorePort.

    Supports dense HNSW search, tsvector full-text search, and
    Reciprocal Rank Fusion for hybrid retrieval.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        """Initialise with async session factory and application settings.

        Args:
            session_factory: SQLAlchemy async session factory.
            settings: Application settings (provides top_k defaults and DB URL).
        """
        self._session_factory = session_factory
        self._settings = settings

    async def upsert(self, chunks: list[ChunkWithEmbedding]) -> int:
        """Insert or update chunks in the vector store.

        Uses ``INSERT … ON CONFLICT (id) DO UPDATE`` for idempotent writes.

        Args:
            chunks: Chunks with precomputed embeddings to persist.

        Returns:
            Number of chunks processed.
        """
        if not chunks:
            return 0

        rows = [
            {
                "id": c.id,
                "document_id": c.document_id,
                "content": c.content,
                "position": c.position,
                "token_count": c.token_count,
                "embedding": c.embedding,
            }
            for c in chunks
        ]
        stmt = pg_insert(ChunkORM).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "content": stmt.excluded.content,
                "position": stmt.excluded.position,
                "token_count": stmt.excluded.token_count,
                "embedding": stmt.excluded.embedding,
            },
        )
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()

        return len(chunks)

    @traced("vector_store.search")
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform HNSW ANN dense search using cosine distance.

        Args:
            query_vector: Query embedding vector (1024-dim).
            top_k: Maximum number of results to return.
            filters: Optional key-value filters; ``"language"`` filters by
                the parent document's ISO 639-1 language code.

        Returns:
            List of retrieved chunks sorted by descending cosine similarity.
        """
        start = time.perf_counter()

        distance_expr = ChunkORM.embedding.cosine_distance(query_vector)
        stmt = (
            select(ChunkORM, DocumentORM.source_path, distance_expr.label("distance"))
            .join(DocumentORM, ChunkORM.document_id == DocumentORM.id)
            .order_by(distance_expr)
            .limit(top_k)
        )
        if filters and filters.get("language"):
            stmt = stmt.where(DocumentORM.language == filters["language"])

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.all()

        observe_histogram(
            "vector_search_duration_seconds",
            time.perf_counter() - start,
            {"search_type": "dense"},
        )
        inc_counter("vector_search_results_count", {"search_type": "dense"})

        return [
            RetrievedChunk(
                chunk_id=chunk_orm.id,
                document_id=chunk_orm.document_id,
                content=chunk_orm.content,
                score=max(0.0, 1.0 - float(distance)),
                source_path=source_path,
                metadata={},
            )
            for chunk_orm, source_path, distance in rows
        ]

    async def _sparse_search(
        self,
        query_text: str,
        top_k: int,
        language: str | None,
    ) -> list[tuple[UUID, float]]:
        """Run PostgreSQL tsvector full-text search (BM25-ish via ts_rank).

        Uses ``plainto_tsquery`` with the ``simple`` dictionary so no stemming
        is applied — consistent across all document languages.

        Args:
            query_text: Raw text to search.
            top_k: Maximum rows to return.
            language: Optional ISO 639-1 code; when set, restricts results to
                chunks whose parent document matches that language.

        Returns:
            List of ``(chunk_id, ts_rank_score)`` pairs ordered by descending rank.
        """
        if language:
            raw_sql = text(
                """
                SELECT c.id,
                       ts_rank(
                           to_tsvector('simple', c.content),
                           plainto_tsquery('simple', :query)
                       ) AS rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE to_tsvector('simple', c.content)
                      @@ plainto_tsquery('simple', :query)
                  AND d.language = :lang
                ORDER BY rank DESC
                LIMIT :limit
                """
            )
            params: dict[str, object] = {
                "query": query_text,
                "limit": top_k,
                "lang": language,
            }
        else:
            raw_sql = text(
                """
                SELECT id,
                       ts_rank(
                           to_tsvector('simple', content),
                           plainto_tsquery('simple', :query)
                       ) AS rank
                FROM chunks
                WHERE to_tsvector('simple', content)
                      @@ plainto_tsquery('simple', :query)
                ORDER BY rank DESC
                LIMIT :limit
                """
            )
            params = {"query": query_text, "limit": top_k}

        async with self._session_factory() as session:
            result = await session.execute(raw_sql, params)
            return [(UUID(str(row.id)), float(row.rank)) for row in result]

    async def _fetch_chunks_by_ids(
        self, chunk_ids: list[UUID]
    ) -> dict[UUID, RetrievedChunk]:
        """Fetch full chunk rows for a set of IDs (used to fill RRF gaps).

        Args:
            chunk_ids: Chunk UUIDs to fetch.

        Returns:
            Mapping of ``chunk_id → RetrievedChunk`` (score set to 0.0).
        """
        if not chunk_ids:
            return {}

        stmt = (
            select(ChunkORM, DocumentORM.source_path)
            .join(DocumentORM, ChunkORM.document_id == DocumentORM.id)
            .where(ChunkORM.id.in_(chunk_ids))
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.all()

        return {
            chunk_orm.id: RetrievedChunk(
                chunk_id=chunk_orm.id,
                document_id=chunk_orm.document_id,
                content=chunk_orm.content,
                score=0.0,
                source_path=source_path,
                metadata={},
            )
            for chunk_orm, source_path in rows
        }

    @traced("vector_store.hybrid_search")
    async def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense + sparse search fused via Reciprocal Rank Fusion.

        Runs HNSW cosine-distance search and tsvector FTS independently,
        then combines scores using RRF (k=60):
        ``score = 1 / (60 + rank_dense) + 1 / (60 + rank_sparse)``.

        Args:
            query_vector: Dense query embedding (1024-dim).
            query_text: Raw query text for full-text search.
            top_k: Final number of chunks to return after fusion.
            filters: Optional key-value filters; ``"language"`` key supported.

        Returns:
            Re-ranked list of retrieved chunks with RRF score.
        """
        start = time.perf_counter()
        vs = self._settings.vector_store
        language: str | None = (filters or {}).get("language")

        dense_chunks = await self.search(
            query_vector=query_vector,
            top_k=vs.top_k_dense,
            filters=filters,
        )
        sparse_pairs = await self._sparse_search(
            query_text=query_text,
            top_k=vs.top_k_sparse,
            language=language,
        )

        rrf_scores: dict[UUID, float] = {}
        chunk_data: dict[UUID, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_chunks, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            chunk_data[cid] = chunk

        missing_ids = {cid for cid, _ in sparse_pairs} - set(chunk_data)
        if missing_ids:
            fetched = await self._fetch_chunks_by_ids(list(missing_ids))
            chunk_data.update(fetched)

        for rank, (cid, _) in enumerate(sparse_pairs, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        observe_histogram(
            "vector_search_duration_seconds",
            time.perf_counter() - start,
            {"search_type": "hybrid"},
        )
        inc_counter("vector_search_results_count", {"search_type": "hybrid"})

        return [
            RetrievedChunk(
                chunk_id=cid,
                document_id=chunk_data[cid].document_id,
                content=chunk_data[cid].content,
                score=rrf_score,
                source_path=chunk_data[cid].source_path,
                metadata=chunk_data[cid].metadata,
            )
            for cid, rrf_score in ranked
            if cid in chunk_data
        ]

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all chunks belonging to a document.

        Args:
            document_id: UUID of the parent document.

        Returns:
            Number of chunks deleted.
        """
        stmt = delete(ChunkORM).where(ChunkORM.document_id == document_id)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            await session.commit()
        return result.rowcount or 0

    async def delete(self, chunk_ids: list[UUID]) -> None:
        """Delete specific chunks by ID.

        Args:
            chunk_ids: UUIDs of chunks to delete.
        """
        if not chunk_ids:
            return
        stmt = delete(ChunkORM).where(ChunkORM.id.in_(chunk_ids))
        async with self._session_factory() as session:
            await session.execute(stmt)
            await session.commit()

    async def count(self) -> int:
        """Return total number of chunks stored.

        Returns:
            Total chunk count.
        """
        stmt = select(func.count()).select_from(ChunkORM)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one()
