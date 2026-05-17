pytestmark = pytest.mark.integration


from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.domain.ports.dto import GenerationRequest
from src.infrastructure.llm.errors import LLMError
from src.infrastructure.llm.gemini import GeminiLLM

_PATCH_TARGET = "src.infrastructure.llm.gemini.ChatGoogleGenerativeAI"


@pytest.fixture()
def settings() -> Settings:
    return Settings()


def _ok_response(text: str = "Hello!", input_tokens: int = 5, output_tokens: int = 10) -> MagicMock:
    from langchain_core.messages import AIMessage

    resp = AIMessage(content=text)
    resp.usage_metadata = {  # type: ignore[assignment]
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return resp  # type: ignore[return-value]


def _make_gemini(mock_cls: MagicMock, settings: Settings) -> tuple[GeminiLLM, MagicMock]:
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    gemini = GeminiLLM(settings)
    return gemini, mock_client


async def test_generate_returns_result(settings: Settings) -> None:
    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(return_value=_ok_response())
        result = await gemini.generate(request)

    assert result.text == "Hello!"
    assert result.input_tokens == 5
    assert result.output_tokens == 10
    assert result.model == settings.llm.model


async def test_generate_with_system_prompt(settings: Settings) -> None:
    request = GenerationRequest(
        messages=[{"role": "user", "content": "Hi"}],
        system_prompt="You are helpful.",
    )

    captured: list[list] = []

    async def mock_ainvoke(msgs: list, **kwargs: object) -> object:
        captured.append(msgs)
        return _ok_response("ok")

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        await gemini.generate(request)

    msgs = captured[0]
    from langchain_core.messages import SystemMessage

    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "You are helpful."


async def test_generate_assistant_role_maps_to_ai_message(settings: Settings) -> None:
    request = GenerationRequest(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Bye"},
        ]
    )

    captured: list[list] = []

    async def mock_ainvoke(msgs: list, **kwargs: object) -> object:
        captured.append(msgs)
        return _ok_response()

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        await gemini.generate(request)

    from langchain_core.messages import AIMessage, HumanMessage

    assert isinstance(captured[0][0], HumanMessage)
    assert isinstance(captured[0][1], AIMessage)
    assert isinstance(captured[0][2], HumanMessage)


async def test_generate_wraps_provider_exception(settings: Settings) -> None:
    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=RuntimeError("provider boom"))
        with pytest.raises(LLMError) as exc_info:
            await gemini.generate(request)

    assert exc_info.value.provider == "gemini"
    assert exc_info.value.original is not None


async def test_generate_does_not_double_wrap_llm_error(settings: Settings) -> None:
    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])
    original_err = LLMError("already wrapped", provider="gemini")

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=original_err)
        with pytest.raises(LLMError) as exc_info:
            await gemini.generate(request)

    assert exc_info.value is original_err


async def test_generate_no_usage_metadata_defaults_to_zero(settings: Settings) -> None:
    from langchain_core.messages import AIMessage

    resp = AIMessage(content="text")
    request = GenerationRequest(messages=[{"role": "user", "content": "q"}])

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(return_value=resp)
        result = await gemini.generate(request)

    assert result.input_tokens == 0
    assert result.output_tokens == 0


async def test_stream_yields_tokens(settings: Settings) -> None:
    from langchain_core.messages import AIMessageChunk

    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    async def mock_astream(*args: object, **kwargs: object) -> AsyncMock:
        for chunk_text in ["Hello", " ", "world"]:
            yield AIMessageChunk(content=chunk_text)

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.astream = mock_astream
        iterator = await gemini.stream(request)
        tokens = [tok async for tok in iterator]

    assert tokens == ["Hello", " ", "world"]


async def test_stream_skips_empty_chunks(settings: Settings) -> None:
    from langchain_core.messages import AIMessageChunk

    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    async def mock_astream(*args: object, **kwargs: object) -> AsyncMock:
        yield AIMessageChunk(content="")
        yield AIMessageChunk(content="text")
        yield AIMessageChunk(content="")

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.astream = mock_astream
        iterator = await gemini.stream(request)
        tokens = [tok async for tok in iterator]

    assert tokens == ["text"]


async def test_stream_wraps_provider_exception(settings: Settings) -> None:
    from langchain_core.messages import AIMessageChunk

    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    async def mock_astream(*args: object, **kwargs: object) -> AsyncMock:
        yield AIMessageChunk(content="partial")
        raise RuntimeError("stream error")

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.astream = mock_astream
        iterator = await gemini.stream(request)
        with pytest.raises(LLMError):
            async for _ in iterator:
                pass


async def test_generate_records_histogram(settings: Settings) -> None:
    from src.shared.metrics import get_metrics_output

    request = GenerationRequest(messages=[{"role": "user", "content": "ping"}])

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(return_value=_ok_response())
        await gemini.generate(request)

    output = get_metrics_output()
    assert "llm_request_duration_seconds" in output
    assert 'provider="gemini"' in output


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Integration: requires real GEMINI_API_KEY env var",
)
async def test_integration_live_generate(settings: Settings) -> None:
    llm = GeminiLLM(settings)
    request = GenerationRequest(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        temperature=0.0,
        max_tokens=10,
    )
    result = await llm.generate(request)
    assert result.text
    assert len(result.text) < 50