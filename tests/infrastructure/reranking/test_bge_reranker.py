from __future__ import annotations

import os
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.config.settings import Settings
from src.domain.ports.dto import RerankRequest, RetrievedChunk
from src.infrastructure.reranking.bge_reranker import BGEReranker


def _chunk(content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        score=score,
        source_path="test.pdf",
    )


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def reranker(settings: Settings) -> BGEReranker:
    return BGEReranker(settings)


def test_model_not_loaded_at_init(reranker: BGEReranker) -> None:
    assert reranker._model is None


async def test_rerank_sort_order(reranker: BGEReranker) -> None:
    chunks = [
        _chunk("low relevance"),
        _chunk("high relevance"),
        _chunk("medium relevance"),
    ]
    request = RerankRequest(query="test query", chunks=chunks, top_k=3)
    mock_model = MagicMock()
    # All scores above the 0.3 threshold — only sort order is exercised here.
    mock_model.compute_score.return_value = [0.4, 0.9, 0.6]
    reranker._model = mock_model

    result = await reranker.rerank(request)

    assert len(result) == 3
    assert result[0].content == "high relevance"
    assert result[1].content == "medium relevance"
    assert result[2].content == "low relevance"
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.6)


async def test_rerank_score_threshold_filters_low_relevance(reranker: BGEReranker) -> None:
    chunks = [_chunk("irrelevant doc"), _chunk("relevant doc")]
    # Score 0.1 is below the default 0.3 threshold; 0.7 passes.
    request = RerankRequest(query="query", chunks=chunks, top_k=5)
    mock_model = MagicMock()
    mock_model.compute_score.return_value = [0.1, 0.7]
    reranker._model = mock_model

    result = await reranker.rerank(request)

    assert len(result) == 1
    assert result[0].score == pytest.approx(0.7)
    assert result[0].content == "relevant doc"


async def test_rerank_all_below_threshold_returns_empty(reranker: BGEReranker) -> None:
    chunks = [_chunk("a"), _chunk("b")]
    request = RerankRequest(query="query", chunks=chunks, top_k=5)
    mock_model = MagicMock()
    mock_model.compute_score.return_value = [0.1, 0.2]
    reranker._model = mock_model

    result = await reranker.rerank(request)

    assert result == []
    assert result[2].score == pytest.approx(0.4)


async def test_rerank_top_k_truncation(reranker: BGEReranker) -> None:
    chunks = [_chunk(f"doc {i}") for i in range(5)]
    request = RerankRequest(query="query", chunks=chunks, top_k=2)
    mock_model = MagicMock()
    mock_model.compute_score.return_value = [0.1, 0.8, 0.3, 0.9, 0.5]
    reranker._model = mock_model

    result = await reranker.rerank(request)

    assert len(result) == 2
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.8)


async def test_rerank_empty_chunks(reranker: BGEReranker) -> None:
    request = RerankRequest(query="query", chunks=[], top_k=5)
    mock_model = MagicMock()
    reranker._model = mock_model

    result = await reranker.rerank(request)

    assert result == []
    mock_model.compute_score.assert_not_called()


async def test_rerank_single_pair_scalar_score(reranker: BGEReranker) -> None:
    chunks = [_chunk("only doc")]
    request = RerankRequest(query="q", chunks=chunks, top_k=1)
    mock_model = MagicMock()
    mock_model.compute_score.return_value = 0.75
    reranker._model = mock_model

    result = await reranker.rerank(request)

    assert len(result) == 1
    assert result[0].score == pytest.approx(0.75)


async def test_rerank_passes_batch_size_to_model(reranker: BGEReranker) -> None:
    chunks = [_chunk("a"), _chunk("b")]
    request = RerankRequest(query="q", chunks=chunks, top_k=2)
    mock_model = MagicMock()
    mock_model.compute_score.return_value = [0.4, 0.6]
    reranker._model = mock_model

    await reranker.rerank(request)

    call_kwargs = mock_model.compute_score.call_args
    assert call_kwargs.kwargs.get("batch_size") == reranker._settings.reranker.batch_size


async def test_rerank_scores_updated_on_returned_chunks(reranker: BGEReranker) -> None:
    original_score = 0.5
    chunks = [_chunk("doc a", score=original_score)]
    request = RerankRequest(query="q", chunks=chunks, top_k=1)
    mock_model = MagicMock()
    mock_model.compute_score.return_value = [0.88]
    reranker._model = mock_model

    result = await reranker.rerank(request)

    assert result[0].score == pytest.approx(0.88)
    assert result[0].score != original_score


@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"),
    reason="Integration: requires FlagEmbedding + model download (set RUN_INTEGRATION=1)",
)
async def test_integration_multilingual_rerank(settings: Settings) -> None:
    from FlagEmbedding import FlagReranker

    reranker = BGEReranker(settings)
    chunks = [
        _chunk("1337 is a coding school in Morocco."),
        _chunk("The weather today is sunny and warm."),
        _chunk("مدرسة 1337 للبرمجة في المغرب."),
    ]
    request = RerankRequest(
        query="What is 1337 coding school?",
        chunks=chunks,
        top_k=2,
    )

    result = await reranker.rerank(request)

    assert len(result) == 2
    assert result[0].score >= result[1].score
    assert reranker._model is not None
    assert isinstance(reranker._model, FlagReranker)
