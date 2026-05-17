from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.prompts import PromptTemplateLoader
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.domain.entities.session import Message, MessageRole
from src.domain.ports.dto import GenerationResult, RetrievedChunk
from src.shared.metrics import REGISTRY


@pytest.fixture()
def loader() -> PromptTemplateLoader:
    return PromptTemplateLoader()


@pytest.fixture()
def session_id() -> uuid.UUID:
    return uuid.uuid4()


def _chunk(chunk_id: str | None = None, page: int | None = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.UUID(chunk_id) if chunk_id else uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="42 is the answer.",
        score=0.9,
        source_path="/kb/handbook.pdf",
        metadata={"page": page} if page is not None else {},
    )


def _make_llm(
    text: str = "Yes [{cid}].",
    input_tokens: int = 7,
    output_tokens: int = 13,
) -> MagicMock:
    mock = MagicMock()
    mock.generate = AsyncMock(
        return_value=GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model="mock-llm",
        )
    )
    return mock


def _counter_value(name: str, **labels: str) -> float:
    val = REGISTRY.get_sample_value(f"{name}_total", labels)
    return float(val) if val is not None else 0.0


def _histogram_count(name: str) -> float:
    val = REGISTRY.get_sample_value(f"{name}_count")
    return float(val) if val is not None else 0.0


async def test_empty_chunks_returns_refusal_without_llm_call(
    loader: PromptTemplateLoader, session_id: uuid.UUID
) -> None:
    llm = _make_llm()
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    answer = await uc.execute(query="What is 42?", retrieved_chunks=[], history=[], language="en")

    assert "don't have enough information" in answer.text.lower()
    assert answer.citations == []
    assert answer.tokens_in == 0
    assert answer.tokens_out == 0
    assert answer.language == "en"
    llm.generate.assert_not_called()


async def test_empty_chunks_returns_french_refusal_for_fr(
    loader: PromptTemplateLoader,
) -> None:
    llm = _make_llm()
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    answer = await uc.execute(
        query="C'est quoi 42 ?", retrieved_chunks=[], history=[], language="fr"
    )

    assert "informations suffisantes" in answer.text
    llm.generate.assert_not_called()


async def test_empty_chunks_returns_arabic_refusal_for_ar(loader: PromptTemplateLoader) -> None:
    llm = _make_llm()
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    answer = await uc.execute(query="42 ؟", retrieved_chunks=[], history=[], language="ar")

    assert "كافية" in answer.text


async def test_unknown_language_falls_back_to_english(loader: PromptTemplateLoader) -> None:
    llm = _make_llm()
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    answer = await uc.execute(query="x", retrieved_chunks=[], history=[], language="zz")

    assert "don't have enough information" in answer.text.lower()


async def test_citations_are_grounded_only(loader: PromptTemplateLoader) -> None:
    chunk = _chunk()
    cid = str(chunk.chunk_id)
    llm = _make_llm(text=f"Yes, the answer is 42 [{cid}] and also [fake-id-not-retrieved].")
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    answer = await uc.execute(
        query="What is the answer?",
        retrieved_chunks=[chunk],
        history=[],
        language="en",
    )

    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == cid
    assert answer.citations[0].source == "/kb/handbook.pdf"
    assert answer.citations[0].page == 1
    assert answer.citations[0].marker == f"[{cid}]"
    llm.generate.assert_called_once()


async def test_duplicate_markers_deduplicated(loader: PromptTemplateLoader) -> None:
    chunk = _chunk()
    cid = str(chunk.chunk_id)
    llm = _make_llm(text=f"42 [{cid}]. Again [{cid}].")
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    answer = await uc.execute(query="q", retrieved_chunks=[chunk], history=[], language="en")

    assert len(answer.citations) == 1


async def test_french_language_renders_french_in_prompt(loader: PromptTemplateLoader) -> None:
    chunk = _chunk()
    cid = str(chunk.chunk_id)
    llm = _make_llm(text=f"La réponse est 42 [{cid}].")
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    await uc.execute(query="C'est quoi 42 ?", retrieved_chunks=[chunk], history=[], language="fr")

    request = llm.generate.await_args.args[0]
    rendered = request.messages[0]["content"]
    assert 'language code "fr"' in rendered
    assert "Je ne dispose pas d'informations suffisantes" in rendered


async def test_history_is_rendered_in_prompt(loader: PromptTemplateLoader) -> None:
    chunk = _chunk()
    cid = str(chunk.chunk_id)
    sid = uuid.uuid4()
    history = [
        Message(session_id=sid, role=MessageRole.USER, content="hello"),
        Message(session_id=sid, role=MessageRole.ASSISTANT, content="hi there"),
    ]
    llm = _make_llm(text=f"42 [{cid}].")
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    await uc.execute(query="follow-up", retrieved_chunks=[chunk], history=history, language="en")

    rendered = llm.generate.await_args.args[0].messages[0]["content"]
    assert "user: hello" in rendered
    assert "assistant: hi there" in rendered


async def test_tokens_recorded_from_llm_result(loader: PromptTemplateLoader) -> None:
    chunk = _chunk()
    cid = str(chunk.chunk_id)
    llm = _make_llm(text=f"42 [{cid}].", input_tokens=11, output_tokens=22)
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    answer = await uc.execute(query="q", retrieved_chunks=[chunk], history=[], language="en")

    assert answer.tokens_in == 11
    assert answer.tokens_out == 22


async def test_metrics_emitted(loader: PromptTemplateLoader) -> None:
    chunk = _chunk()
    cid = str(chunk.chunk_id)
    llm = _make_llm(text=f"42 [{cid}].")
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    in_before = _counter_value("generation_tokens", kind="input")
    out_before = _counter_value("generation_tokens", kind="output")
    hist_before = _histogram_count("generation_citations_count")

    await uc.execute(query="q", retrieved_chunks=[chunk], history=[], language="en")

    assert _counter_value("generation_tokens", kind="input") == pytest.approx(in_before + 1.0)
    assert _counter_value("generation_tokens", kind="output") == pytest.approx(out_before + 1.0)
    assert _histogram_count("generation_citations_count") == pytest.approx(hist_before + 1.0)


async def test_llm_exception_propagates(loader: PromptTemplateLoader) -> None:
    chunk = _chunk()
    llm = MagicMock()
    llm.generate = AsyncMock(side_effect=RuntimeError("boom"))
    uc = GenerateAnswerUseCase(llm=llm, prompt_template_loader=loader)

    with pytest.raises(RuntimeError, match="boom"):
        await uc.execute(query="q", retrieved_chunks=[chunk], history=[], language="en")
