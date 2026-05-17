"""Gemini LLM adapter — implements LLMPort via langchain-google-genai."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import Settings
from src.domain.ports.dto import GenerationRequest, GenerationResult
from src.domain.ports.tracer import TracerPort
from src.infrastructure.llm._base import BaseLLM
from src.infrastructure.llm.errors import (
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

__all__ = ["GeminiLLM"]

log = structlog.get_logger(__name__)

_RATE_LIMIT_CODE = 429


class GeminiLLM(BaseLLM):
    """Gemini LLM adapter implementing LLMPort via langchain-google-genai.

    Uses ``ChatGoogleGenerativeAI`` (not the deprecated ``google.generativeai``
    package) and delegates retry, metrics, and error-wrapping to ``BaseLLM``.
    Optionally emits GenAI observability spans via a ``TracerPort``.
    """

    PROVIDER = "gemini"

    def __init__(
        self,
        settings: Settings,
        tracer: TracerPort | None = None,
    ) -> None:
        """Initialize Gemini adapter with settings and an optional tracer.

        Args:
            settings: Application settings providing LLM configuration.
            tracer: Optional tracing adapter for GenAI observability spans.
        """
        super().__init__(settings)
        self._tracer = tracer
        api_key = settings.llm.api_key.get_secret_value()
        self._client: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
            model=settings.llm.model,
            google_api_key=api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    def _should_retry(self, exc: BaseException) -> bool:
        """Retry on Gemini 5xx errors, rate-limit (429), and HTTP timeouts.

        Args:
            exc: Exception to evaluate.

        Returns:
            True if the call should be retried, False otherwise.
        """
        import httpx
        from google.genai.errors import ClientError, ServerError

        if isinstance(exc, ServerError):
            return True
        if isinstance(exc, ClientError) and exc.code == _RATE_LIMIT_CODE:
            return True
        return bool(isinstance(exc, httpx.TimeoutException))

    def _wrap_exception(self, exc: Exception) -> LLMError:
        """Map a Gemini provider exception to a specific LLMError subclass.

        Args:
            exc: Provider exception to map.

        Returns:
            A specific ``LLMError`` subclass matching the error type.
        """
        import httpx
        from google.genai.errors import ClientError, ServerError

        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(str(exc), provider=self.PROVIDER, original=exc)
        if isinstance(exc, ServerError):
            return LLMProviderError(str(exc), provider=self.PROVIDER, original=exc)
        if isinstance(exc, ClientError) and exc.code == _RATE_LIMIT_CODE:
            return LLMRateLimitError(str(exc), provider=self.PROVIDER, original=exc)
        return LLMError(str(exc), provider=self.PROVIDER, original=exc)

    def _build_messages(self, request: GenerationRequest) -> list[BaseMessage]:
        """Convert GenerationRequest to a list of LangChain message objects.

        Args:
            request: Generation request with messages and optional system prompt.

        Returns:
            List of ``BaseMessage`` objects ready for model invocation.
        """
        messages: list[BaseMessage] = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    async def _ainvoke_with_retry(self, messages: list[BaseMessage]) -> BaseMessage:
        """Invoke the underlying LLM client with retry on transient 429 errors.

        Gemini free-tier RPM quotas trigger 429 RESOURCE_EXHAUSTED. We retry up
        to three times with exponential backoff (8s, 16s, 32s) so a brief burst
        of requests is smoothed out instead of failing the request outright.
        Non-rate-limit errors are re-raised immediately.
        """
        delays = (8.0, 16.0, 32.0)
        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0, *delays)):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await self._client.ainvoke(messages)
            except Exception as exc:
                msg = str(exc)
                is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                last_exc = exc
                if not is_rate_limit or attempt == len(delays):
                    raise
                log.warning(
                    "gemini.rate_limit_retry",
                    attempt=attempt + 1,
                    next_delay_s=delays[attempt] if attempt < len(delays) else None,
                )
        assert last_exc is not None
        raise last_exc

    async def _do_generate(self, request: GenerationRequest) -> GenerationResult:
        """Invoke ChatGoogleGenerativeAI and return a structured GenerationResult.

        Emits a GenAI observability span (if tracer provided) with attributes:
        ``gen_ai.system``, ``gen_ai.request.model``, ``gen_ai.usage.input_tokens``,
        ``gen_ai.usage.output_tokens``.

        Args:
            request: Generation request.

        Returns:
            GenerationResult with text content and token usage metadata.
        """
        messages = self._build_messages(request)
        ctx = (
            self._tracer.start_span(
                "gemini.generate",
                input={
                    "gen_ai.system": "gemini",
                    "gen_ai.request.model": self._settings.llm.model,
                },
            )
            if self._tracer
            else None
        )

        _err: Exception | None = None
        try:
            response: BaseMessage = await self._ainvoke_with_retry(messages)
        except Exception as exc:
            _err = exc
            raise
        finally:
            if _err is not None and self._tracer is not None and ctx is not None:
                self._tracer.end_span(ctx, error=_err)

        text = response.content if isinstance(response.content, str) else str(response.content)
        input_tokens = 0
        output_tokens = 0
        if isinstance(response, AIMessage) and response.usage_metadata is not None:
            input_tokens = response.usage_metadata["input_tokens"]
            output_tokens = response.usage_metadata["output_tokens"]

        log.debug(
            "gemini.generate_complete",
            model=self._settings.llm.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if self._tracer is not None and ctx is not None:
            self._tracer.end_span(
                ctx,
                output={
                    "gen_ai.usage.input_tokens": input_tokens,
                    "gen_ai.usage.output_tokens": output_tokens,
                },
            )

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._settings.llm.model,
        )

    def _do_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Return an async generator that streams tokens from Gemini.

        Args:
            request: Generation request.

        Returns:
            Async iterator yielding text tokens as they arrive.
        """
        messages = self._build_messages(request)

        async def _gen() -> AsyncGenerator[str, None]:
            async for chunk in self._client.astream(messages):
                content: Any = chunk.content
                if isinstance(content, str) and content:
                    yield content

        return _gen()
