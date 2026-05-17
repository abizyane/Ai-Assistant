import pytest
from typer.testing import CliRunner

from src.interface.cli.main import app

runner = CliRunner()


@pytest.fixture()
def mock_ingest_use_case(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.domain.ports.dto import IngestionReport

    report = IngestionReport(
        files_processed=2,
        files_skipped=1,
        chunks_created=10,
        duration_seconds=1.23,
        errors=[],
    )
    uc = MagicMock()
    uc.execute = AsyncMock(return_value=report)
    monkeypatch.setattr("src.interface.cli.main.build_ingest_use_case", lambda s=None: uc)
    monkeypatch.setattr("src.interface.cli.main.build_settings", lambda: MagicMock())
    return uc


@pytest.fixture()
def mock_agent(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.domain.entities.answer import AnswerWithCitations

    answer = AnswerWithCitations(
        text="42",
        citations=[],
        language="en",
        tokens_in=10,
        tokens_out=5,
    )
    state = {"final_answer": answer}
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=state)
    monkeypatch.setattr("src.interface.cli.main.build_agent", lambda s=None: graph)
    monkeypatch.setattr("src.interface.cli.main.build_settings", lambda: MagicMock())
    return graph


@pytest.fixture()
def mock_health_ok(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
    conn_cm.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.connect = MagicMock(return_value=conn_cm)
    monkeypatch.setattr("src.interface.cli.main.build_engine", lambda s=None: engine)
    return engine


@pytest.fixture()
def mock_session_repo(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    repo = MagicMock()
    repo.list_sessions = AsyncMock(
        return_value=[
            {
                "id": "01KRNJ2MNPN1DMJPFNQV41K4SZ",
                "user_id": None,
                "created_at": "2026-05-16 10:00:00",
                "last_active": "2026-05-16 11:00:00",
                "message_count": 3,
            }
        ]
    )
    repo.get_history = AsyncMock(return_value=[])
    monkeypatch.setattr("src.interface.cli.main.build_session_repo", lambda s=None: repo)
    return repo


def test_app_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "chat" in result.output
    assert "evaluate" in result.output
    assert "sessions" in result.output
    assert "health" in result.output


def test_ingest_exits_nonzero_for_nonexistent_path(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr("src.interface.cli.main.build_settings", lambda: MagicMock())
    monkeypatch.setattr(
        "src.interface.cli.main.build_ingest_use_case", lambda s=None: MagicMock()
    )
    result = runner.invoke(app, ["ingest", "/nonexistent/path/abc123"])
    assert result.exit_code != 0


def test_ingest_success(tmp_path, mock_ingest_use_case):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 mock")
    result = runner.invoke(app, ["ingest", str(tmp_path)])
    assert result.exit_code == 0
    assert "Ingestion Summary" in result.output


def test_chat_single_query(mock_agent):
    result = runner.invoke(app, ["chat", "What is 1337?"])
    assert result.exit_code == 0
    assert "42" in result.output


def test_health_success(mock_health_ok):
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert "healthy" in result.output.lower() or "reachable" in result.output.lower()


def test_health_failure_exits_nonzero(monkeypatch):
    from unittest.mock import MagicMock

    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("DB down")
    monkeypatch.setattr("src.interface.cli.main.build_engine", lambda s=None: engine)
    result = runner.invoke(app, ["health"])
    assert result.exit_code != 0


def test_sessions_list(mock_session_repo):
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 0
    assert "01KRNJ2MNPN1DMJPFNQV41K4SZ" in result.output


def test_sessions_show_empty(mock_session_repo):
    result = runner.invoke(app, ["sessions", "show", "01KRNJ2MNPN1DMJPFNQV41K4SZ"])
    assert result.exit_code == 0
    assert "No messages" in result.output


def test_evaluate_help():
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "dataset" in result.output.lower() or "DATASET" in result.output
