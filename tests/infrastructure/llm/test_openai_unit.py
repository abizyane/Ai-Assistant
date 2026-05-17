from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config.settings import Settings
from src.domain.ports.dto import GenerationRequest
from src.infrastructure.llm.errors import (
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

_PATCH_TARGET = "src.infrastructure.llm.openai.ChatOpenAI"


@pytest.fixture()
def settings() -> Settings:
    return Settings()


def _make_openai(mock_cls: MagicMock, settings: Settings) -> tuple[object, MagicMock]:
    from src.infrastructure.llm.openai import OpenAILLM

    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    llm = OpenAILLM(settings)
    return llm, mock_client


def _ok_response(text: str = "hello") -> AIMessage:
    msg = AIMessage(content=text)
    msg.usage_metadata = {"input_tokens": 5, "output_tokens": 3}  # type: ignore[assignment]
    return msg


def test_build_messages_with_system_prompt(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        req = GenerationRequest(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are helpful.",
        )
        msgs = llm._build_messages(req)  # type: ignore[attr-defined]
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)


def test_build_messages_assistant_role(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        req = GenerationRequest(
            messages=[{"role": "assistant", "content": "sure"}],
        )
        msgs = llm._build_messages(req)  # type: ignore[attr-defined]
    assert isinstance(msgs[0], AIMessage)


async def test_do_generate_returns_result(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(return_value=_ok_response("world"))
        req = GenerationRequest(messages=[{"role": "user", "content": "q"}])
        result = await llm._do_generate(req)  # type: ignore[attr-defined]
    assert result.text == "world"
    assert result.input_tokens == 5
    assert result.output_tokens == 3


async def test_do_generate_no_usage_metadata(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)
        msg = AIMessage(content="ok")
        msg.usage_metadata = None  # type: ignore[assignment]
        mock_client.ainvoke = AsyncMock(return_value=msg)
        req = GenerationRequest(messages=[{"role": "user", "content": "q"}])
        result = await llm._do_generate(req)  # type: ignore[attr-defined]
    assert result.text == "ok"
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_wrap_exception_timeout(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        exc = openai.APITimeoutError(request=MagicMock())
        wrapped = llm._wrap_exception(exc)  # type: ignore[attr-defined]
    assert isinstance(wrapped, LLMTimeoutError)


def test_wrap_exception_rate_limit(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        exc = openai.RateLimitError("rate", response=MagicMock(), body=None)
        wrapped = llm._wrap_exception(exc)  # type: ignore[attr-defined]
    assert isinstance(wrapped, LLMRateLimitError)


def test_wrap_exception_internal_server(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        exc = openai.InternalServerError("err", response=MagicMock(), body=None)
        wrapped = llm._wrap_exception(exc)  # type: ignore[attr-defined]
    assert isinstance(wrapped, LLMProviderError)


def test_wrap_exception_generic(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        exc = Exception("unknown")
        wrapped = llm._wrap_exception(exc)  # type: ignore[attr-defined]
    assert isinstance(wrapped, LLMError)


def test_should_retry_rate_limit(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        exc = openai.RateLimitError("rate", response=MagicMock(), body=None)
        assert llm._should_retry(exc) is True  # type: ignore[attr-defined]


def test_should_retry_internal_server(settings: Settings) -> None:
    import openai

    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        exc = openai.InternalServerError("err", response=MagicMock(), body=None)
        assert llm._should_retry(exc) is True  # type: ignore[attr-defined]


def test_should_not_retry_generic(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        llm, _ = _make_openai(mock_cls, settings)
        assert llm._should_retry(ValueError("nope")) is False  # type: ignore[attr-defined]


async def test_do_stream_yields_tokens(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        llm, mock_client = _make_openai(mock_cls, settings)

        async def _fake_astream(*_a: object, **_kw: object) -> object:
            for token in ["Hello", " world"]:
                yield AIMessage(content=token)

        mock_client.astream = _fake_astream
        req = GenerationRequest(messages=[{"role": "user", "content": "q"}])
        tokens = [t async for t in llm._do_stream(req)]  # type: ignore[attr-defined]
    assert tokens == ["Hello", " world"]
