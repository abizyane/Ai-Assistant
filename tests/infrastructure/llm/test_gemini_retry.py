from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.domain.ports.dto import GenerationRequest
from src.infrastructure.llm.errors import LLMError, LLMProviderError, LLMRateLimitError
from src.infrastructure.llm.gemini import GeminiLLM

_PATCH_TARGET = "src.infrastructure.llm.gemini.ChatGoogleGenerativeAI"


@pytest.fixture()
def settings() -> Settings:
    return Settings()


def _ok_response(text: str = "ok") -> MagicMock:
    from langchain_core.messages import AIMessage

    resp = AIMessage(content=text)
    resp.usage_metadata = None  # type: ignore[assignment]
    return resp  # type: ignore[return-value]


def _make_gemini(mock_cls: MagicMock, settings: Settings) -> tuple[GeminiLLM, MagicMock]:
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    gemini = GeminiLLM(settings)
    return gemini, mock_client


async def test_retry_on_rate_limit_then_success(settings: Settings) -> None:
    from google.genai.errors import ClientError

    request = GenerationRequest(messages=[{"role": "user", "content": "test"}])
    rate_limit_err = ClientError(429, {"error": {"message": "Rate limit exceeded", "code": 429}})
    ok = _ok_response("done")

    call_count = 0

    async def side_effect(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise rate_limit_err
        return ok

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=side_effect)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gemini.generate(request)

    assert result.text == "done"
    assert call_count == 2


async def test_retry_on_server_error_then_success(settings: Settings) -> None:
    from google.genai.errors import ServerError

    request = GenerationRequest(messages=[{"role": "user", "content": "test"}])
    server_err = ServerError(503, {"error": {"message": "Service Unavailable", "code": 503}})
    ok = _ok_response("recovered")

    side_effects: list[object] = [server_err, server_err, ok]

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=side_effects)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gemini.generate(request)

    assert result.text == "recovered"


async def test_max_retries_exceeded_raises_llm_provider_error(settings: Settings) -> None:
    from google.genai.errors import ServerError

    request = GenerationRequest(messages=[{"role": "user", "content": "test"}])
    server_err = ServerError(503, {"error": {"message": "Unavailable", "code": 503}})

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=server_err)
        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(LLMProviderError) as exc_info,
        ):
            await gemini.generate(request)

    assert exc_info.value.provider == "gemini"
    assert exc_info.value.original is server_err


async def test_max_retries_exceeded_rate_limit_raises_llm_rate_limit_error(
    settings: Settings,
) -> None:
    from google.genai.errors import ClientError

    request = GenerationRequest(messages=[{"role": "user", "content": "test"}])
    rate_err = ClientError(429, {"error": {"message": "Rate limit", "code": 429}})

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=rate_err)
        with patch("asyncio.sleep", new_callable=AsyncMock), pytest.raises(LLMRateLimitError):
            await gemini.generate(request)


async def test_timeout_error_wraps_to_llm_timeout_error(settings: Settings) -> None:
    import httpx

    from src.infrastructure.llm.errors import LLMTimeoutError

    request = GenerationRequest(messages=[{"role": "user", "content": "test"}])
    timeout_err = httpx.ConnectTimeout("timed out")

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=timeout_err)
        with patch("asyncio.sleep", new_callable=AsyncMock), pytest.raises(LLMTimeoutError):
            await gemini.generate(request)


async def test_should_retry_returns_false_for_unknown_error(settings: Settings) -> None:
    with patch(_PATCH_TARGET) as mock_cls:
        gemini, _ = _make_gemini(mock_cls, settings)
    assert gemini._should_retry(RuntimeError("unrelated")) is False


async def test_should_retry_returns_true_for_server_error(settings: Settings) -> None:
    from google.genai.errors import ServerError

    err = ServerError(503, {"error": {"message": "oops", "code": 503}})
    with patch(_PATCH_TARGET) as mock_cls:
        gemini, _ = _make_gemini(mock_cls, settings)
    assert gemini._should_retry(err) is True


async def test_should_retry_returns_true_for_rate_limit(settings: Settings) -> None:
    from google.genai.errors import ClientError

    err = ClientError(429, {"error": {"message": "rate limit", "code": 429}})
    with patch(_PATCH_TARGET) as mock_cls:
        gemini, _ = _make_gemini(mock_cls, settings)
    assert gemini._should_retry(err) is True


async def test_should_retry_returns_false_for_auth_error(settings: Settings) -> None:
    from google.genai.errors import ClientError

    err = ClientError(401, {"error": {"message": "unauthorized", "code": 401}})
    with patch(_PATCH_TARGET) as mock_cls:
        gemini, _ = _make_gemini(mock_cls, settings)
    assert gemini._should_retry(err) is False


async def test_retry_count_from_settings(settings: Settings) -> None:
    assert settings.llm.max_retries == 3


async def test_non_retryable_error_raises_immediately(settings: Settings) -> None:
    from google.genai.errors import ClientError

    request = GenerationRequest(messages=[{"role": "user", "content": "test"}])
    auth_err = ClientError(401, {"error": {"message": "Unauthorized", "code": 401}})

    call_count = 0

    async def side_effect(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        raise auth_err

    with patch(_PATCH_TARGET) as mock_cls:
        gemini, mock_client = _make_gemini(mock_cls, settings)
        mock_client.ainvoke = AsyncMock(side_effect=side_effect)
        with patch("asyncio.sleep", new_callable=AsyncMock), pytest.raises(LLMError):
            await gemini.generate(request)

    assert call_count == 1
