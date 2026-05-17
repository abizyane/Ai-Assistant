from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.use_cases.retrieve import RetrieveUseCase
from src.domain.ports.dto import RetrievedChunk


def _chunk(score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Test content",
        score=score,
        source_path="/kb/doc.pdf",
    )


def _make_settings(top_k_dense: int = 20, top_k_rerank: int = 5) -> MagicMock:
    settings = MagicMock()
    settings.vector_store.top_k_dense = top_k_dense
    settings.vector_store.top_k_rerank = top_k_rerank
    return settings


def _make_use_case(
    chunks: list[RetrievedChunk] | None = None,
    reranked: list[RetrievedChunk] | None = None,
    top_k_dense: int = 20,
    top_k_rerank: int = 5,
) -> tuple[RetrieveUseCase, MagicMock, MagicMock, MagicMock]:
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)

    vector_store = MagicMock()
    vector_store.hybrid_search = AsyncMock(return_value=chunks if chunks is not None else [])

    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=reranked if reranked is not None else [])

    settings = _make_settings(top_k_dense=top_k_dense, top_k_rerank=top_k_rerank)

    uc = RetrieveUseCase(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        settings=settings,
    )
    return uc, embedder, vector_store, reranker


async def test_happy_path_returns_reranked_chunks() -> None:
    raw_chunks = [_chunk() for _ in range(10)]
    reranked = [_chunk(score=0.95) for _ in range(5)]

    uc, embedder, vector_store, reranker = _make_use_case(
        chunks=raw_chunks, reranked=reranked, top_k_rerank=5
    )

    result = await uc.execute("What is 1337?", language="en")

    assert result == reranked
    embedder.embed_query.assert_called_once_with("What is 1337?")
    vector_store.hybrid_search.assert_called_once()
    reranker.rerank.assert_called_once()


async def test_language_auto_detected_when_not_provided() -> None:
    raw_chunks = [_chunk() for _ in range(10)]
    reranked = [_chunk() for _ in range(5)]

    uc, _, vector_store, _ = _make_use_case(chunks=raw_chunks, reranked=reranked, top_k_rerank=5)

    with patch("src.application.use_cases.retrieve.detect", return_value="fr") as mock_detect:
        result = await uc.execute("Qu'est-ce que 1337?")

    mock_detect.assert_called_once_with("Qu'est-ce que 1337?")
    call_kwargs = vector_store.hybrid_search.call_args
    assert call_kwargs.kwargs["filters"] == {"language": "fr"}
    assert len(result) == 5


async def test_language_passed_explicitly_bypasses_detection() -> None:
    uc, _, vector_store, _ = _make_use_case(chunks=[_chunk()], top_k_rerank=5)

    with patch("src.application.use_cases.retrieve.detect") as mock_detect:
        await uc.execute("What is 42?", language="en")

    mock_detect.assert_not_called()
    call_kwargs = vector_store.hybrid_search.call_args
    assert call_kwargs.kwargs["filters"] == {"language": "en"}


async def test_reranker_skipped_when_chunks_le_top_k_rerank() -> None:
    raw_chunks = [_chunk() for _ in range(3)]

    uc, _, _, reranker = _make_use_case(chunks=raw_chunks, top_k_rerank=5)

    result = await uc.execute("Query", language="en")

    reranker.rerank.assert_not_called()
    assert result == raw_chunks


async def test_empty_search_result_returns_empty_list() -> None:
    uc, _, _, reranker = _make_use_case(chunks=[], top_k_rerank=5)

    result = await uc.execute("Query", language="en")

    assert result == []
    reranker.rerank.assert_not_called()


async def test_embedder_failure_propagates() -> None:
    uc, embedder, _, _ = _make_use_case()
    embedder.embed_query = AsyncMock(side_effect=RuntimeError("embed failed"))

    with pytest.raises(RuntimeError, match="embed failed"):
        await uc.execute("Query", language="en")
