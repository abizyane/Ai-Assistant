pytestmark = pytest.mark.integration


"""Tests for SessionRepository — unit (mocked) and integration (skipif no DB)."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from src.domain.entities.session import Message, MessageRole
from src.infrastructure.persistence.session_repo import SessionRepository


def _make_factory(mock_session: AsyncSession) -> MagicMock:
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=ctx)
    return factory


class TestCreateSession:
    async def test_returns_26_char_ulid_string(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()

        repo = SessionRepository(_make_factory(mock_session))
        result = await repo.create_session()

        assert isinstance(result, str)
        assert len(result) == 26

    async def test_adds_orm_and_commits(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()

        repo = SessionRepository(_make_factory(mock_session))
        await repo.create_session(user_id="user-42")

        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()

    async def test_different_calls_return_unique_ids(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()

        repo = SessionRepository(_make_factory(mock_session))
        id1 = await repo.create_session()
        id2 = await repo.create_session()

        assert id1 != id2


class TestAppendMessage:
    async def test_add_execute_commit_called(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()

        session_id = str(ULID())
        repo = SessionRepository(_make_factory(mock_session))
        await repo.append_message(session_id, role="user", content="hello")

        mock_session.add.assert_called_once()
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    async def test_increments_metric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()

        captured: list[tuple[str, dict[str, str] | None]] = []
        monkeypatch.setattr(
            "src.infrastructure.persistence.session_repo.inc_counter",
            lambda name, labels=None: captured.append((name, labels)),
        )

        session_id = str(ULID())
        repo = SessionRepository(_make_factory(mock_session))
        await repo.append_message(session_id, role="assistant", content="hi")

        assert any("session_messages_appended_total" in name for name, _ in captured)
        assert any(labels == {"role": "assistant"} for _, labels in captured)

    async def test_metadata_accepted_without_error(self) -> None:
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.add = MagicMock()

        session_id = str(ULID())
        repo = SessionRepository(_make_factory(mock_session))
        await repo.append_message(
            session_id,
            role="user",
            content="test",
            metadata={"key": "value"},
        )
        mock_session.commit.assert_awaited_once()


class TestGetHistory:
    async def test_returns_message_entities_in_order(self) -> None:
        session_uuid = uuid.uuid4()
        now = datetime.now(UTC)

        row_a = MagicMock()
        row_a.id = uuid.uuid4()
        row_a.session_id = session_uuid
        row_a.role = "user"
        row_a.content = "first"
        row_a.created_at = now

        row_b = MagicMock()
        row_b.id = uuid.uuid4()
        row_b.session_id = session_uuid
        row_b.role = "assistant"
        row_b.content = "second"
        row_b.created_at = now

        scalars = MagicMock()
        scalars.all.return_value = [row_a, row_b]
        result = MagicMock()
        result.scalars.return_value = scalars

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=result)

        session_id = str(ULID.from_uuid(session_uuid))
        repo = SessionRepository(_make_factory(mock_session))
        messages = await repo.get_history(session_id)

        assert len(messages) == 2
        assert all(isinstance(m, Message) for m in messages)
        assert messages[0].content == "first"
        assert messages[0].role == MessageRole.USER
        assert messages[1].content == "second"
        assert messages[1].role == MessageRole.ASSISTANT

    async def test_empty_returns_empty_list(self) -> None:
        session_uuid = uuid.uuid4()
        scalars = MagicMock()
        scalars.all.return_value = []
        result = MagicMock()
        result.scalars.return_value = scalars

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=result)

        session_id = str(ULID.from_uuid(session_uuid))
        repo = SessionRepository(_make_factory(mock_session))
        messages = await repo.get_history(session_id, limit=5)

        assert messages == []
        mock_session.execute.assert_awaited_once()


class TestListSessions:
    async def test_returns_list_of_summary_dicts(self) -> None:
        now = datetime.now(UTC)
        sess_uuid = uuid.uuid4()

        row = MagicMock()
        row.id = sess_uuid
        row.user_id = "user-1"
        row.created_at = now
        row.last_active = now
        row.message_count = 3

        result = MagicMock()
        result.all.return_value = [row]

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=result)

        repo = SessionRepository(_make_factory(mock_session))
        sessions = await repo.list_sessions()

        assert len(sessions) == 1
        s = sessions[0]
        assert s["user_id"] == "user-1"
        assert s["message_count"] == 3
        assert len(s["id"]) == 26
        assert s["created_at"] == now

    async def test_filters_by_user_id_when_given(self) -> None:
        result = MagicMock()
        result.all.return_value = []

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(return_value=result)

        repo = SessionRepository(_make_factory(mock_session))
        sessions = await repo.list_sessions(user_id="user-99")

        assert sessions == []
        mock_session.execute.assert_awaited_once()


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Integration: requires TEST_DATABASE_URL env var pointing to live Postgres",
)
class TestSessionRepositoryIntegration:
    async def test_create_append_and_get_history_roundtrip(self) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine

        from src.infrastructure.persistence.engine import create_session_factory
        from src.infrastructure.persistence.models import Base

        engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = create_session_factory(engine)
        repo = SessionRepository(factory)

        session_id = await repo.create_session(user_id="integration-user")
        assert len(session_id) == 26

        await repo.append_message(session_id, role="user", content="hello")
        await repo.append_message(session_id, role="assistant", content="world")

        messages = await repo.get_history(session_id)
        assert len(messages) == 2
        assert messages[0].content == "hello"
        assert messages[1].content == "world"

        sessions = await repo.list_sessions(user_id="integration-user")
        assert any(s["id"] == session_id for s in sessions)
        assert sessions[0]["message_count"] == 2

        await engine.dispose()