pytestmark = pytest.mark.integration


from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.domain.ports.dto import GenerationRequest
from src.infrastructure.llm.errors import LLMError
from src.infrastructure.llm.openai import OpenAILLM

_PATCH_TARGET = "src.infrastructure.llm.openai.ChatOpenAI"


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


def _make_openai(mock_cls: MagicMock, settings: Settings) -> tuple[OpenAILLM, MagicMock]:
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    llm = OpenAILLM(settings)
    return llm, mock_client


async def test_generate_returns_result(settings: Settings) -> None:
    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(return_value=_ok_response())
        result = await llm.generate(request)

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
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        await llm.generate(request)

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
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        await llm.generate(request)

    from langchain_core.messages import AIMessage, HumanMessage

    assert isinstance(captured[0][0], HumanMessage)
    assert isinstance(captured[0][1], AIMessage)
    assert isinstance(captured[0][2], HumanMessage)


async def test_generate_wraps_provider_exception(settings: Settings) -> None:
    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=RuntimeError("provider boom"))
        with pytest.raises(LLMError) as exc_info:
            await llm.generate(request)

    assert exc_info.value.provider == "openai"
    assert exc_info.value.original is not None


async def test_generate_does_not_double_wrap_llm_error(settings: Settings) -> None:
    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])
    original_err = LLMError("already wrapped", provider="openai")

    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=original_err)
        with pytest.raises(LLMError) as exc_info:
            await llm.generate(request)

    assert exc_info.value is original_err


async def test_generate_no_usage_metadata_defaults_to_zero(settings: Settings) -> None:
    from langchain_core.messages import AIMessage

    resp = AIMessage(content="text")
    request = GenerationRequest(messages=[{"role": "user", "content": "q"}])

    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(return_value=resp)
        result = await llm.generate(request)

    assert result.input_tokens == 0
    assert result.output_tokens == 0


async def test_stream_yields_tokens(settings: Settings) -> None:
    from langchain_core.messages import AIMessageChunk

    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    async def mock_astream(*args: object, **kwargs: object) -> AsyncMock:
        for chunk_text in ["Hello", " ", "world"]:
            yield AIMessageChunk(content=chunk_text)

    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.astream = mock_astream
        iterator = await llm.stream(request)
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
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.astream = mock_astream
        iterator = await llm.stream(request)
        tokens = [tok async for tok in iterator]

    assert tokens == ["text"]


async def test_stream_wraps_provider_exception(settings: Settings) -> None:
    from langchain_core.messages import AIMessageChunk

    request = GenerationRequest(messages=[{"role": "user", "content": "Hi"}])

    async def mock_astream(*args: object, **kwargs: object) -> AsyncMock:
        yield AIMessageChunk(content="partial")
        raise RuntimeError("stream error")

    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.astream = mock_astream
        iterator = await llm.stream(request)
        with pytest.raises(LLMError):
            async for _ in iterator:
                pass


async def test_generate_records_histogram(settings: Settings) -> None:
    from src.shared.metrics import get_metrics_output

    request = GenerationRequest(messages=[{"role": "user", "content": "ping"}])

    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(return_value=_ok_response())
        await llm.generate(request)

    output = get_metrics_output()
    assert "llm_request_duration_seconds" in output
    assert 'provider="openai"' in output


async def test_should_retry_on_internal_server_error(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)

    mock_response = MagicMock()
    mock_response.status_code = 500
    exc = openai.InternalServerError("server error", response=mock_response, body=None)
    assert llm._should_retry(exc) is True


async def test_should_retry_on_rate_limit_error(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)

    mock_response = MagicMock()
    mock_response.status_code = 429
    exc = openai.RateLimitError("rate limit", response=mock_response, body=None)
    assert llm._should_retry(exc) is True


async def test_should_retry_on_timeout_error(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)

    exc = openai.APITimeoutError(request=MagicMock())
    assert llm._should_retry(exc) is True


async def test_wrap_exception_timeout(settings: Settings) -> None:
    import openai

    from src.infrastructure.llm.errors import LLMTimeoutError

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)

    exc = openai.APITimeoutError(request=MagicMock())
    wrapped = llm._wrap_exception(exc)
    assert isinstance(wrapped, LLMTimeoutError)
    assert wrapped.provider == "openai"


async def test_wrap_exception_provider_error(settings: Settings) -> None:
    import openai

    from src.infrastructure.llm.errors import LLMProviderError

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)

    mock_response = MagicMock()
    mock_response.status_code = 500
    exc = openai.InternalServerError("server error", response=mock_response, body=None)
    wrapped = llm._wrap_exception(exc)
    assert isinstance(wrapped, LLMProviderError)
    assert wrapped.provider == "openai"


async def test_wrap_exception_rate_limit(settings: Settings) -> None:
    import openai

    from src.infrastructure.llm.errors import LLMRateLimitError

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)

    mock_response = MagicMock()
    mock_response.status_code = 429
    exc = openai.RateLimitError("rate limit", response=mock_response, body=None)
    wrapped = llm._wrap_exception(exc)
    assert isinstance(wrapped, LLMRateLimitError)
    assert wrapped.provider == "openai"


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Integration: requires real OPENAI_API_KEY env var",
)
async def test_integration_live_generate(settings: Settings) -> None:
    llm = OpenAILLM(settings)
    request = GenerationRequest(
        messages=[{"role": "user", "content": "Say hi in one word."}],
        temperature=0.0,
        max_tokens=10,
    )
    result = await llm.generate(request)
    assert result.text
    assert len(result.text) < 50