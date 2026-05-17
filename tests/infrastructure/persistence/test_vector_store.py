pytestmark = pytest.mark.integration


"""Unit and integration tests for PGVectorStore."""

from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Settings
from src.domain.ports.dto import ChunkWithEmbedding, RetrievedChunk
from src.infrastructure.persistence.vector_store import PGVectorStore

_DIM = 1024


def _make_vector() -> list[float]:
    return [0.1] * _DIM


def _make_chunk(doc_id: uuid.UUID | None = None) -> ChunkWithEmbedding:
    return ChunkWithEmbedding(
        id=uuid.uuid4(),
        document_id=doc_id or uuid.uuid4(),
        content="test content",
        embedding=_make_vector(),
        position=0,
        token_count=10,
        source_path="/docs/test.pdf",
    )


def _make_retrieved_chunk(chunk_id: uuid.UUID | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="some content",
        score=0.9,
        source_path="/docs/doc.pdf",
    )


def _make_store(factory: MagicMock) -> PGVectorStore:
    settings = Settings(llm={"api_key": "test"})  # type: ignore[arg-type]
    return PGVectorStore(factory, settings)


def _mock_session_factory() -> tuple[MagicMock, AsyncMock]:
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=mock_cm), mock_session


async def test_upsert_empty_returns_zero() -> None:
    factory, _ = _mock_session_factory()
    store = _make_store(factory)
    assert await store.upsert([]) == 0
    factory.assert_not_called()


async def test_upsert_single_chunk_returns_count() -> None:
    factory, mock_session = _mock_session_factory()
    mock_session.execute = AsyncMock()
    store = _make_store(factory)

    result = await store.upsert([_make_chunk()])

    assert result == 1
    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


async def test_upsert_multiple_chunks() -> None:
    factory, mock_session = _mock_session_factory()
    mock_session.execute = AsyncMock()
    store = _make_store(factory)

    result = await store.upsert([_make_chunk() for _ in range(5)])

    assert result == 5
    mock_session.execute.assert_awaited_once()


async def test_search_returns_mocked_results() -> None:
    factory, mock_session = _mock_session_factory()
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    mock_orm = MagicMock()
    mock_orm.id = chunk_id
    mock_orm.document_id = doc_id
    mock_orm.content = "hello world"

    mock_result = MagicMock()
    mock_result.all.return_value = [(mock_orm, "/path/doc.pdf", 0.2)]
    mock_session.execute = AsyncMock(return_value=mock_result)

    store = _make_store(factory)
    results = await store.search(query_vector=_make_vector(), top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == chunk_id
    assert results[0].document_id == doc_id
    assert results[0].content == "hello world"
    assert results[0].source_path == "/path/doc.pdf"
    assert abs(results[0].score - 0.8) < 1e-6


async def test_search_empty_result() -> None:
    factory, mock_session = _mock_session_factory()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_result)

    store = _make_store(factory)
    assert await store.search(query_vector=_make_vector(), top_k=10) == []


async def test_search_score_clamped_to_zero_for_large_distance() -> None:
    factory, mock_session = _mock_session_factory()
    mock_orm = MagicMock()
    mock_orm.id = uuid.uuid4()
    mock_orm.document_id = uuid.uuid4()
    mock_orm.content = "far away"

    mock_result = MagicMock()
    mock_result.all.return_value = [(mock_orm, "/path/doc.pdf", 1.5)]
    mock_session.execute = AsyncMock(return_value=mock_result)

    store = _make_store(factory)
    results = await store.search(query_vector=_make_vector(), top_k=1)
    assert results[0].score == 0.0


async def test_hybrid_search_combines_dense_and_sparse() -> None:
    factory, _ = _mock_session_factory()
    store = _make_store(factory)
    chunk_a = _make_retrieved_chunk()
    chunk_b = _make_retrieved_chunk()

    store.search = AsyncMock(return_value=[chunk_a])  # type: ignore[method-assign]
    store._sparse_search = AsyncMock(return_value=[(chunk_b.chunk_id, 0.8)])  # type: ignore[method-assign]
    store._fetch_chunks_by_ids = AsyncMock(return_value={chunk_b.chunk_id: chunk_b})  # type: ignore[method-assign]

    results = await store.hybrid_search(
        query_vector=_make_vector(), query_text="test query", top_k=10
    )

    assert len(results) == 2
    store.search.assert_awaited_once()
    store._sparse_search.assert_awaited_once()


async def test_hybrid_search_rrf_scores_are_positive() -> None:
    factory, _ = _mock_session_factory()
    store = _make_store(factory)
    chunk = _make_retrieved_chunk()

    store.search = AsyncMock(return_value=[chunk])  # type: ignore[method-assign]
    store._sparse_search = AsyncMock(return_value=[(chunk.chunk_id, 0.9)])  # type: ignore[method-assign]
    store._fetch_chunks_by_ids = AsyncMock(return_value={})  # type: ignore[method-assign]

    results = await store.hybrid_search(
        query_vector=_make_vector(), query_text="test", top_k=5
    )

    assert len(results) == 1
    assert results[0].score > 0.0


async def test_hybrid_search_respects_top_k() -> None:
    factory, _ = _mock_session_factory()
    store = _make_store(factory)
    chunks = [_make_retrieved_chunk() for _ in range(10)]
    sparse_pairs = [(c.chunk_id, float(i)) for i, c in enumerate(chunks)]

    store.search = AsyncMock(return_value=chunks)  # type: ignore[method-assign]
    store._sparse_search = AsyncMock(return_value=sparse_pairs)  # type: ignore[method-assign]
    store._fetch_chunks_by_ids = AsyncMock(return_value={})  # type: ignore[method-assign]

    results = await store.hybrid_search(
        query_vector=_make_vector(), query_text="query", top_k=3
    )
    assert len(results) <= 3


async def test_hybrid_search_with_language_filter() -> None:
    factory, _ = _mock_session_factory()
    store = _make_store(factory)
    chunk = _make_retrieved_chunk()

    store.search = AsyncMock(return_value=[chunk])  # type: ignore[method-assign]
    store._sparse_search = AsyncMock(return_value=[])  # type: ignore[method-assign]
    store._fetch_chunks_by_ids = AsyncMock(return_value={})  # type: ignore[method-assign]

    results = await store.hybrid_search(
        query_vector=_make_vector(),
        query_text="query",
        top_k=5,
        filters={"language": "fr"},
    )

    store.search.assert_awaited_once_with(
        query_vector=_make_vector(),
        top_k=store._settings.vector_store.top_k_dense,
        filters={"language": "fr"},
    )
    store._sparse_search.assert_awaited_once_with(
        query_text="query",
        top_k=store._settings.vector_store.top_k_sparse,
        language="fr",
    )
    assert len(results) == 1


async def test_delete_by_document_returns_rowcount() -> None:
    factory, mock_session = _mock_session_factory()
    mock_result = MagicMock()
    mock_result.rowcount = 3
    mock_session.execute = AsyncMock(return_value=mock_result)

    store = _make_store(factory)
    assert await store.delete_by_document(uuid.uuid4()) == 3
    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


async def test_delete_by_document_none_rowcount_returns_zero() -> None:
    factory, mock_session = _mock_session_factory()
    mock_result = MagicMock()
    mock_result.rowcount = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    store = _make_store(factory)
    assert await store.delete_by_document(uuid.uuid4()) == 0


async def test_delete_empty_list_is_noop() -> None:
    factory, _ = _mock_session_factory()
    store = _make_store(factory)
    await store.delete([])
    factory.assert_not_called()


async def test_delete_calls_execute() -> None:
    factory, mock_session = _mock_session_factory()
    mock_session.execute = AsyncMock()

    store = _make_store(factory)
    await store.delete([uuid.uuid4(), uuid.uuid4()])

    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


async def test_count_returns_scalar() -> None:
    factory, mock_session = _mock_session_factory()
    mock_result = MagicMock()
    mock_result.scalar_one.return_value = 42
    mock_session.execute = AsyncMock(return_value=mock_result)

    store = _make_store(factory)
    assert await store.count() == 42


_SKIP_INTEGRATION = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"),
    reason="RUN_INTEGRATION not set — skipping integration tests",
)


@_SKIP_INTEGRATION
async def test_integration_upsert_and_count() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from src.infrastructure.persistence.engine import create_session_factory

    settings = Settings()
    engine = create_async_engine(settings.vector_store.database_url)
    factory = create_session_factory(engine)
    store = PGVectorStore(factory, settings)

    doc_id = uuid.uuid4()
    chunks = [
        ChunkWithEmbedding(
            id=uuid.uuid4(),
            document_id=doc_id,
            content=f"integration test chunk {i}",
            embedding=_make_vector(),
            position=i,
            token_count=5,
            source_path="/integration/test.pdf",
        )
        for i in range(3)
    ]

    initial_count = await store.count()
    inserted = await store.upsert(chunks)
    assert inserted == 3
    assert await store.count() == initial_count + 3

    await store.delete_by_document(doc_id)
    assert await store.count() == initial_count

    await engine.dispose()


@_SKIP_INTEGRATION
async def test_integration_search_returns_results() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from src.infrastructure.persistence.engine import create_session_factory

    settings = Settings()
    engine = create_async_engine(settings.vector_store.database_url)
    factory = create_session_factory(engine)
    store = PGVectorStore(factory, settings)

    doc_id = uuid.uuid4()
    chunk = ChunkWithEmbedding(
        id=uuid.uuid4(),
        document_id=doc_id,
        content="searchable integration content",
        embedding=_make_vector(),
        position=0,
        token_count=4,
        source_path="/integration/search.pdf",
    )
    await store.upsert([chunk])

    results = await store.search(query_vector=_make_vector(), top_k=5)
    assert len(results) >= 1

    await store.delete_by_document(doc_id)
    await engine.dispose()