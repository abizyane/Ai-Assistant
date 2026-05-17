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
        return GenerationResult(
            text=text, input_tokens=1, output_tokens=1, model="mock"
        )

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
    grade_llm = _llm_with(["yes", "yes"])
    graph = build_agent_graph(retrieve_uc, generate_uc, grade_llm, settings)

    final: AgentState = await graph.ainvoke(
        {"query": "What is the capital of France?", "language": "en"}
    )

    assert retrieve_uc.execute.await_count == 1
    assert generate_uc.execute.await_count == 1
    assert final["rewrite_count"] == 0 if "rewrite_count" in final else True
    assert final.get("retry_count", 0) == 0
    assert final["grounded"] is True
    assert final["final_answer"].text == "Paris."


async def test_rewrite_when_irrelevant(
    retrieve_uc: MagicMock, generate_uc: MagicMock, settings: Settings
) -> None:
    grade_llm = _llm_with(
        [
            "no",
            "rewritten question",
            "yes",
            "yes",
        ]
    )
    graph = build_agent_graph(retrieve_uc, generate_uc, grade_llm, settings)

    final: AgentState = await graph.ainvoke(
        {"query": "Tell me about Paris", "language": "en"}
    )

    assert retrieve_uc.execute.await_count == 2
    assert generate_uc.execute.await_count == 1
    assert final["rewrite_count"] == 1
    assert final["rewritten_query"] == "rewritten question"
    assert final["grounded"] is True


async def test_retry_on_ungrounded(
    retrieve_uc: MagicMock, generate_uc: MagicMock, settings: Settings
) -> None:
    grade_llm = _llm_with(
        [
            "yes",
            "no",
            "yes",
        ]
    )
    graph = build_agent_graph(retrieve_uc, generate_uc, grade_llm, settings)

    final: AgentState = await graph.ainvoke(
        {"query": "Capital of France?", "language": "en"}
    )

    assert retrieve_uc.execute.await_count == 1
    assert generate_uc.execute.await_count == 2
    assert final["retry_count"] == 1
    assert final["grounded"] is True


async def test_loop_bounds_respected(
    retrieve_uc: MagicMock, generate_uc: MagicMock, settings: Settings
) -> None:
    grade_llm = MagicMock()

    async def _always(_request: object) -> GenerationResult:
        return GenerationResult(
            text="no", input_tokens=1, output_tokens=1, model="mock"
        )

    grade_llm.generate = AsyncMock(side_effect=_always)

    graph = build_agent_graph(retrieve_uc, generate_uc, grade_llm, settings)

    final: AgentState = await graph.ainvoke(
        {"query": "Anything", "language": "en"},
        config={"recursion_limit": 50},
    )

    assert retrieve_uc.execute.await_count <= 2
    assert generate_uc.execute.await_count <= 2
    assert final.get("rewrite_count", 0) <= 1
    assert final.get("retry_count", 0) <= settings.agent.max_regen_attempts + 1
