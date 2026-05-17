from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.domain.entities.answer import AnswerCitation, AnswerWithCitations
from src.domain.entities.session import Message, MessageRole
from src.domain.ports.dto import IngestionReport
from src.interface.api.main import app


def _make_answer(text: str = "hello world") -> AnswerWithCitations:
    return AnswerWithCitations(
        text=text,
        citations=[],
        language="en",
        tokens_in=10,
        tokens_out=5,
    )


def _make_answer_with_citation() -> AnswerWithCitations:
    return AnswerWithCitations(
        text="see [ref-1]",
        citations=[AnswerCitation(chunk_id="abc", source="doc.pdf", page=1, marker="[ref-1]")],
        language="en",
        tokens_in=20,
        tokens_out=8,
    )


def _make_ingest_report() -> IngestionReport:
    return IngestionReport(
        files_processed=2,
        files_skipped=0,
        chunks_created=42,
        duration_seconds=1.5,
        errors=[],
    )


@pytest.fixture
async def client(monkeypatch):
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"final_answer": _make_answer()})

    mock_ingest_uc = MagicMock()
    mock_ingest_uc.execute = AsyncMock(return_value=_make_ingest_report())

    mock_session_repo = MagicMock()
    mock_session_repo.get_history = AsyncMock(return_value=[])

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.execute = AsyncMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)

    monkeypatch.setattr("src.infrastructure.di.build_agent", lambda s=None: mock_agent)
    monkeypatch.setattr(
        "src.infrastructure.di.build_ingest_use_case", lambda s=None: mock_ingest_uc
    )
    monkeypatch.setattr(
        "src.infrastructure.di.build_session_repo", lambda s=None: mock_session_repo
    )
    monkeypatch.setattr("src.infrastructure.di.build_engine", lambda s=None: mock_engine)
    monkeypatch.setattr("src.infrastructure.di.build_embedder", lambda s=None: MagicMock())
    monkeypatch.setattr("src.infrastructure.di.build_reranker", lambda s=None: MagicMock())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "db" in body
    assert "langfuse" in body


async def test_health_db_fail(monkeypatch):
    mock_engine = MagicMock()
    mock_engine.connect = MagicMock(side_effect=RuntimeError("db down"))
    mock_engine.dispose = AsyncMock(return_value=None)

    monkeypatch.setattr("src.infrastructure.di.build_agent", lambda s=None: MagicMock())
    monkeypatch.setattr(
        "src.infrastructure.di.build_ingest_use_case", lambda s=None: MagicMock()
    )
    monkeypatch.setattr(
        "src.infrastructure.di.build_session_repo", lambda s=None: MagicMock()
    )
    monkeypatch.setattr("src.infrastructure.di.build_engine", lambda s=None: mock_engine)
    monkeypatch.setattr("src.infrastructure.di.build_embedder", lambda s=None: MagicMock())
    monkeypatch.setattr("src.infrastructure.di.build_reranker", lambda s=None: MagicMock())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["db"] == "fail"
    assert body["status"] == "degraded"


async def test_sync_chat_returns_answer(client):
    resp = await client.post("/chat/sync", json={"query": "What is 1337?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hello world"
    assert "citations" in body
    assert "tokens_in" in body
    assert "tokens_out" in body


async def test_sse_chat_streams_data_events(client):
    chunks: list[str] = []
    async with client.stream("POST", "/chat", json={"query": "Tell me about 1337"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        async for chunk in resp.aiter_text():
            if chunk.strip():
                chunks.append(chunk)

    data_lines = [c for c in chunks if c.startswith("data:")]
    assert len(data_lines) >= 1


async def test_ingest_returns_summary(client):
    resp = await client.post("/ingest", json={"path": "/some/path"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["files"] == 2
    assert body["chunks"] == 42
    assert isinstance(body["document_ids"], list)


async def test_sessions_returns_empty_history(client):
    resp = await client.get("/sessions/01KRNJ2MNPN1DMJPFNQV41K4SZ")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "01KRNJ2MNPN1DMJPFNQV41K4SZ"
    assert body["messages"] == []


async def test_sessions_with_messages(monkeypatch):
    msg = Message(
        id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        role=MessageRole.USER,
        content="hello",
        created_at=datetime.now(UTC),
    )
    mock_repo = MagicMock()
    mock_repo.get_history = AsyncMock(return_value=[msg])
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.execute = AsyncMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)

    monkeypatch.setattr("src.infrastructure.di.build_agent", lambda s=None: MagicMock())
    monkeypatch.setattr(
        "src.infrastructure.di.build_ingest_use_case", lambda s=None: MagicMock()
    )
    monkeypatch.setattr("src.infrastructure.di.build_session_repo", lambda s=None: mock_repo)
    monkeypatch.setattr("src.infrastructure.di.build_engine", lambda s=None: mock_engine)
    monkeypatch.setattr("src.infrastructure.di.build_embedder", lambda s=None: MagicMock())
    monkeypatch.setattr("src.infrastructure.di.build_reranker", lambda s=None: MagicMock())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.get("/sessions/01KRNJ2MNPN1DMJPFNQV41K4SZ")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello"


async def test_metrics_endpoint_returns_prometheus_format(client):
    resp = await client.get("/metrics/", follow_redirects=True)
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "text/plain" in content_type or "openmetrics" in content_type


async def test_sync_chat_agent_error(monkeypatch):
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"final_answer": None})
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_conn = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_conn.execute = AsyncMock()
    mock_engine.connect = MagicMock(return_value=mock_conn)

    monkeypatch.setattr("src.infrastructure.di.build_agent", lambda s=None: mock_agent)
    monkeypatch.setattr(
        "src.infrastructure.di.build_ingest_use_case", lambda s=None: MagicMock()
    )
    monkeypatch.setattr(
        "src.infrastructure.di.build_session_repo", lambda s=None: MagicMock()
    )
    monkeypatch.setattr("src.infrastructure.di.build_engine", lambda s=None: mock_engine)
    monkeypatch.setattr("src.infrastructure.di.build_embedder", lambda s=None: MagicMock())
    monkeypatch.setattr("src.infrastructure.di.build_reranker", lambda s=None: MagicMock())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.post("/chat/sync", json={"query": "test"})

    assert resp.status_code in (500, 422)
