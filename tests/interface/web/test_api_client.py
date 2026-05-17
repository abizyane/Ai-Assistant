from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.interface.web.api_client import RAGAPIClient


@pytest.fixture()
def client() -> RAGAPIClient:
    return RAGAPIClient(base_url="http://test", langfuse_base_url="http://langfuse")


def test_langfuse_trace_url(client: RAGAPIClient) -> None:
    assert client.langfuse_trace_url("abc123") == "http://langfuse/trace/abc123"


def test_langfuse_trace_url_strips_trailing_slash() -> None:
    c = RAGAPIClient(base_url="http://api/", langfuse_base_url="http://lf/")
    assert c.langfuse_trace_url("x") == "http://lf/trace/x"


async def test_health_returns_json(client: RAGAPIClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.health()

    assert result == {"status": "ok"}


async def test_chat_sync_sends_payload(client: RAGAPIClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "hello", "citations": []}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.chat_sync("hi", session_id="s1", language="en")

    assert result["text"] == "hello"
    call_kwargs = mock_async_client.__aenter__.return_value.post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["query"] == "hi"
    assert payload["session_id"] == "s1"
    assert payload["language"] == "en"


async def test_chat_stream_async_yields_tokens(client: RAGAPIClient) -> None:
    lines = [
        'data: {"token": "Hello"}',
        'data: {"token": " world"}',
        'data: {"done": true, "citations": []}',
    ]

    async def _aiter_lines() -> object:
        for line in lines:
            yield line

    mock_stream_response = MagicMock()
    mock_stream_response.raise_for_status = MagicMock()
    mock_stream_response.aiter_lines = _aiter_lines
    mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
    mock_stream_response.__aexit__ = AsyncMock(return_value=False)

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.stream = MagicMock(return_value=mock_stream_response)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        events = [e async for e in client.chat_stream_async("hi")]

    assert events[0] == {"token": "Hello"}
    assert events[1] == {"token": " world"}
    assert events[2] == {"done": True, "citations": []}


def test_chat_stream_sync_collects_tokens(client: RAGAPIClient) -> None:
    lines = [
        'data: {"token": "A"}',
        'data: {"token": "B"}',
        'data: {"done": true, "citations": [{"source": "doc.pdf"}]}',
    ]

    async def _aiter_lines() -> object:
        for line in lines:
            yield line

    mock_stream_response = MagicMock()
    mock_stream_response.raise_for_status = MagicMock()
    mock_stream_response.aiter_lines = _aiter_lines
    mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
    mock_stream_response.__aexit__ = AsyncMock(return_value=False)

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.stream = MagicMock(return_value=mock_stream_response)

    citations_out: list[object] = []
    with patch("httpx.AsyncClient", return_value=mock_async_client):
        tokens = list(client.chat_stream_sync("hi", citations_out=citations_out))

    assert tokens == ["A", "B"]
    assert citations_out == [{"source": "doc.pdf"}]


async def test_ingest_sends_path(client: RAGAPIClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "files_processed": 1,
        "chunks_created": 5,
        "duration_seconds": 0.5,
    }
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.ingest("/tmp/docs", language_hint="fr")

    assert result["files_processed"] == 1
    call_kwargs = mock_async_client.__aenter__.return_value.post.call_args
    assert call_kwargs.kwargs["json"]["path"] == "/tmp/docs"
    assert call_kwargs.kwargs["json"]["language_hint"] == "fr"


async def test_get_session_returns_messages(client: RAGAPIClient) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"session_id": "s1", "messages": []}
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_async_client):
        result = await client.get_session("s1")

    assert result["session_id"] == "s1"


def test_chat_stream_sync_raises_on_producer_error(client: RAGAPIClient) -> None:
    async def _bad_stream(*_a: object, **_kw: object) -> object:
        raise RuntimeError("backend down")
        yield  # make it an async generator

    with (
        patch.object(client, "chat_stream_async", side_effect=_bad_stream),
        pytest.raises(RuntimeError),
    ):
        list(client.chat_stream_sync("hi"))
