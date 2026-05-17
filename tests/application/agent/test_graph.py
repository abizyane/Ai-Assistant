from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.agent import AgentState, build_agent_graph
from src.config.settings import Settings
from src.domain.entities.answer import AnswerWithCitations
from src.domain.ports.dto import GenerationResult, RetrievedChunk


def _chunk(text: str = "Paris is the capital of France.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        score=0.9,
        source_path="/kb/doc.pdf",
        metadata={"page": 1},
    )


def _answer(text: str = "Paris.") -> AnswerWithCitations:
    return AnswerWithCitations(
        text=text,
        citations=[],
        language="en",
        tokens_in=5,
        tokens_out=3,
    )


def _llm_with(responses: list[str]) -> MagicMock:
    mock = MagicMock()
    queue = list(responses)

    async def _generate(_request: object) -> GenerationResult:
        text = queue.pop(0) if queue else "no"
        return GenerationResult(text=text, input_tokens=1, output_tokens=1, model="mock")

    mock.generate = AsyncMock(side_effect=_generate)
    return mock


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def retrieve_uc() -> MagicMock:
    mock = MagicMock()
    mock.execute = AsyncMock(return_value=[_chunk()])
    return mock


@pytest.fixture()
def generate_uc() -> MagicMock:
    mock = MagicMock()
    mock.execute = AsyncMock(return_value=_answer())
    return mock


async def test_happy_path(
    retrieve_uc: MagicMock, generate_uc: MagicMock, settings: Settings
) -> None:
    grade_llm = _llm_with(["yes"])
    graph = build_agent_graph(retrieve_uc, generate_uc, grade_llm, settings)

    final: AgentState = await graph.ainvoke(
        {"query": "What is the capital of France?", "language": "en"}
    )

    assert retrieve_uc.execute.await_count == 1
    assert generate_uc.execute.await_count == 1
    assert final.get("retry_count", 0) == 0
    assert final["grounded"] is True
    assert final["final_answer"].text == "Paris."


async def test_retry_on_ungrounded(
    retrieve_uc: MagicMock, generate_uc: MagicMock, settings: Settings
) -> None:
    # verify says "no" first (not grounded), then "yes" after retry
    grade_llm = _llm_with(["no", "yes"])
    graph = build_agent_graph(retrieve_uc, generate_uc, grade_llm, settings)

    final: AgentState = await graph.ainvoke({"query": "Capital of France?", "language": "en"})

    assert retrieve_uc.execute.await_count == 1
    assert generate_uc.execute.await_count == 2
    assert final["retry_count"] == 1
    assert final["grounded"] is True


async def test_stops_after_max_retries(
    retrieve_uc: MagicMock, generate_uc: MagicMock, settings: Settings
) -> None:
    # verify always returns "no" — graph should stop and return the last answer
    grade_llm = MagicMock()

    async def _always_no(_request: object) -> GenerationResult:
        return GenerationResult(text="no", input_tokens=1, output_tokens=1, model="mock")

    grade_llm.generate = AsyncMock(side_effect=_always_no)
    graph = build_agent_graph(retrieve_uc, generate_uc, grade_llm, settings)

    final: AgentState = await graph.ainvoke(
        {"query": "Anything", "language": "en"},
        config={"recursion_limit": 50},
    )

    assert generate_uc.execute.await_count <= settings.agent.max_regen_attempts + 1
    assert final.get("retry_count", 0) <= settings.agent.max_regen_attempts + 1
    assert final["final_answer"] is not None
