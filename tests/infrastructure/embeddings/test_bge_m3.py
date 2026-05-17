pytestmark = pytest.mark.integration


from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.settings import Settings
from src.infrastructure.embeddings.bge_m3 import BGEM3Embedder


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def embedder(settings: Settings) -> BGEM3Embedder:
    return BGEM3Embedder(settings)


def test_model_not_loaded_at_init(embedder: BGEM3Embedder) -> None:
    assert embedder._model is None


def test_dimension_property(embedder: BGEM3Embedder) -> None:
    assert embedder.dimension == 1024


async def test_embed_texts_returns_correct_shape(embedder: BGEM3Embedder) -> None:
    texts = ["hello world", "another text"]
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.zeros((2, 1024), dtype=np.float32)}
    embedder._model = mock_model

    result = await embedder.embed_texts(texts)

    assert len(result) == 2
    assert len(result[0]) == 1024
    assert len(result[1]) == 1024


async def test_embed_texts_calls_encode_with_correct_params(embedder: BGEM3Embedder) -> None:
    texts = ["text a", "text b"]
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.zeros((2, 1024), dtype=np.float32)}
    embedder._model = mock_model

    await embedder.embed_texts(texts)

    call_kwargs = mock_model.encode.call_args
    assert call_kwargs.kwargs.get("return_dense") is True
    assert call_kwargs.kwargs.get("return_sparse") is False
    assert call_kwargs.kwargs.get("return_colbert_vecs") is False
    assert call_kwargs.kwargs.get("batch_size") == embedder._settings.embedding.batch_size


async def test_embed_query_returns_single_vector(embedder: BGEM3Embedder) -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.ones((1, 1024), dtype=np.float32)}
    embedder._model = mock_model

    result = await embedder.embed_query("What is 1337?")

    assert isinstance(result, list)
    assert len(result) == 1024
    assert all(isinstance(v, float) for v in result)


async def test_embed_texts_empty_list(embedder: BGEM3Embedder) -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.zeros((0, 1024), dtype=np.float32)}
    embedder._model = mock_model

    result = await embedder.embed_texts([])

    assert result == []


async def test_embed_texts_emits_histogram(embedder: BGEM3Embedder) -> None:
    from src.shared.metrics import get_metrics_output

    mock_model = MagicMock()
    mock_model.encode.return_value = {"dense_vecs": np.zeros((1, 1024), dtype=np.float32)}
    embedder._model = mock_model

    await embedder.embed_texts(["test"])

    output = get_metrics_output()
    assert "embedding_request_duration_seconds" in output


@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"),
    reason="Integration: requires FlagEmbedding + model download (set RUN_INTEGRATION=1)",
)
async def test_integration_embed_texts(settings: Settings) -> None:
    from FlagEmbedding import BGEM3FlagModel

    embedder = BGEM3Embedder(settings)
    texts = ["1337 is a coding school.", "مدرسة 1337 للبرمجة."]
    result = await embedder.embed_texts(texts)

    assert len(result) == 2
    assert len(result[0]) == 1024
    assert embedder._model is not None
    assert isinstance(embedder._model, BGEM3FlagModel)