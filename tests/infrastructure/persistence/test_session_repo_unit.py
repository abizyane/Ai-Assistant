from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.infrastructure.persistence.session_repo import SessionRepository


def _make_repo() -> tuple[SessionRepository, MagicMock]:
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    mock_factory = MagicMock(return_value=mock_db)
    repo = SessionRepository(session_factory=mock_factory)
    return repo, mock_db


async def test_create_session_returns_ulid_string() -> None:
    repo, mock_db = _make_repo()
    session_id = await repo.create_session()
    assert isinstance(session_id, str)
    assert len(session_id) == 26
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


async def test_create_session_with_user_id() -> None:
    repo, mock_db = _make_repo()
    session_id = await repo.create_session(user_id="user-42")
    assert len(session_id) == 26
    added_orm = mock_db.add.call_args[0][0]
    assert added_orm.user_id == "user-42"


async def test_append_message_commits() -> None:
    repo, mock_db = _make_repo()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = MagicMock()
    mock_db.execute.return_value = mock_result

    with patch("src.infrastructure.persistence.session_repo.inc_counter"):
        await repo.append_message(
            session_id="01KRNJ2MNPN1DMJPFNQV41K4SZ",
            role="user",
            content="hello",
        )
    mock_db.commit.assert_called()


async def test_get_history_returns_empty_when_no_session() -> None:
    repo, mock_db = _make_repo()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    history = await repo.get_history("01KRNJ2MNPN1DMJPFNQV41K4SZ")
    assert history == []


async def test_list_sessions_returns_list() -> None:
    repo, mock_db = _make_repo()

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    sessions = await repo.list_sessions()
    assert sessions == []
