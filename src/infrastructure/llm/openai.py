"""OpenAI LLM adapter — implements LLMPort via langchain-openai."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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

__all__ = ["OpenAILLM"]

log = structlog.get_logger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM adapter implementing LLMPort via langchain-openai.

    Uses ``ChatOpenAI`` and delegates retry, metrics, and error-wrapping to
    ``BaseLLM``. Optionally emits GenAI observability spans via a
    ``TracerPort``.
    """

    PROVIDER = "openai"

    def __init__(
        self,
        settings: Settings,
        tracer: TracerPort | None = None,
    ) -> None:
        super().__init__(settings)
        self._tracer = tracer
        api_key = settings.llm.api_key.get_secret_value()
        client_kwargs: dict[str, Any] = {
            "model": settings.llm.model,
            "api_key": api_key,
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
        }
        if settings.llm.base_url:
            client_kwargs["base_url"] = settings.llm.base_url
        self._client: ChatOpenAI = ChatOpenAI(**client_kwargs)

    def _should_retry(self, exc: BaseException) -> bool:
        """Retry on OpenAI 5xx errors, rate-limit (429), and API timeouts."""
        import openai

        if isinstance(exc, openai.InternalServerError):
            return True
        if isinstance(exc, openai.RateLimitError):
            return True
        return isinstance(exc, openai.APITimeoutError)

    def _wrap_exception(self, exc: Exception) -> LLMError:
        import openai

        if isinstance(exc, openai.APITimeoutError):
            return LLMTimeoutError(str(exc), provider=self.PROVIDER, original=exc)
        if isinstance(exc, openai.InternalServerError):
            return LLMProviderError(str(exc), provider=self.PROVIDER, original=exc)
        if isinstance(exc, openai.RateLimitError):
            return LLMRateLimitError(str(exc), provider=self.PROVIDER, original=exc)
        return LLMError(str(exc), provider=self.PROVIDER, original=exc)

    def _build_messages(self, request: GenerationRequest) -> list[BaseMessage]:
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

    async def _do_generate(self, request: GenerationRequest) -> GenerationResult:
        """Invoke ChatOpenAI and return a structured GenerationResult.

        Emits a GenAI observability span (if tracer provided) with attributes:
        ``gen_ai.system``, ``gen_ai.request.model``, ``gen_ai.usage.input_tokens``,
        ``gen_ai.usage.output_tokens``.
        """
        messages = self._build_messages(request)
        ctx = (
            self._tracer.start_span(
                "openai.generate",
                input={
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": self._settings.llm.model,
                },
            )
            if self._tracer
            else None
        )

        _err: Exception | None = None
        try:
            response: BaseMessage = await self._client.ainvoke(messages)
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
            "openai.generate_complete",
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
        messages = self._build_messages(request)

        async def _gen() -> AsyncGenerator[str, None]:
            async for chunk in self._client.astream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content

        return _gen()
