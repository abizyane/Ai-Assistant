"""Abstract base for LLM adapters with retry, metrics, and error wrapping."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import ClassVar

import structlog
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from src.config.settings import Settings
from src.domain.ports.dto import GenerationRequest, GenerationResult
from src.infrastructure.llm.errors import LLMError
from src.shared.metrics import observe_histogram
from src.shared.tracing import traced

__all__ = ["BaseLLM"]

log = structlog.get_logger(__name__)

_HISTOGRAM_NAME = "llm_request_duration_seconds"


class BaseLLM:
    """Abstract base for LLM adapters providing retry, metrics, and error wrapping.

    Subclasses must set ``PROVIDER`` and implement ``_do_generate`` and
    ``_do_stream``. This class handles:

    - Tenacity exponential-backoff retry on transient errors
    - Prometheus latency histogram (``llm_request_duration_seconds``)
    - Uniform wrapping of provider exceptions into ``LLMError`` hierarchy
    """

    PROVIDER: ClassVar[str] = "unknown"

    def __init__(self, settings: Settings) -> None:
        """Store settings; no provider client is created here.

        Args:
            settings: Application settings providing LLM configuration.
        """
        self._settings = settings

    def _should_retry(self, exc: BaseException) -> bool:
        """Return True if the exception is a transient error worth retrying.

        Default implementation never retries. Subclasses override this to add
        provider-specific logic (rate limits, 5xx errors, timeouts).

        Args:
            exc: Exception to evaluate.

        Returns:
            True if the call should be retried, False otherwise.
        """
        return False

    def _wrap_exception(self, exc: Exception) -> LLMError:
        """Map a provider exception to a domain ``LLMError``.

        Override in subclasses to return specific ``LLMError`` subclasses
        (e.g. ``LLMRateLimitError`` for 429, ``LLMTimeoutError`` for timeouts).

        Args:
            exc: Provider exception to wrap.

        Returns:
            An ``LLMError`` wrapping the original exception.
        """
        return LLMError(str(exc), provider=self.PROVIDER, original=exc)

    async def _do_generate(self, request: GenerationRequest) -> GenerationResult:
        """Provider-specific generation logic. Override in subclasses.

        Args:
            request: Generation request.

        Returns:
            Generation result from the provider.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    def _do_stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Provider-specific streaming logic. Override in subclasses.

        Returns a synchronously-constructed async iterator so that
        ``BaseLLM.stream`` can wrap it in a single inner generator.

        Args:
            request: Generation request.

        Returns:
            Async iterator of text tokens.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError

    async def _generate_with_retry(self, request: GenerationRequest) -> GenerationResult:
        """Execute ``_do_generate`` with tenacity exponential-backoff retry.

        Args:
            request: Generation request.

        Returns:
            Generation result from the first successful attempt.

        Raises:
            Exception: Last provider exception after all retries are exhausted.
        """
        last_result: GenerationResult | None = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.llm.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(self._should_retry),
            reraise=True,
        ):
            with attempt:
                last_result = await self._do_generate(request)
        if last_result is None:  # pragma: no cover
            raise LLMError("No result produced", provider=self.PROVIDER)
        return last_result

    @traced("llm.generate")
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a complete response with retry and Prometheus metrics.

        Args:
            request: Generation request including messages and parameters.

        Returns:
            Generation result with text and token usage.

        Raises:
            LLMError: On any provider error after all retries are exhausted.
        """
        start = time.perf_counter()
        try:
            return await self._generate_with_retry(request)
        except LLMError:
            raise
        except Exception as exc:
            raise self._wrap_exception(exc) from exc
        finally:
            observe_histogram(
                _HISTOGRAM_NAME,
                time.perf_counter() - start,
                {
                    "provider": self.PROVIDER,
                    "model": self._settings.llm.model,
                    "operation": "generate",
                },
            )

    @traced("llm.stream")
    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream response tokens with Prometheus metrics.

        The returned ``AsyncIterator`` must be fully consumed to trigger metrics
        recording. Mid-stream provider exceptions are wrapped into ``LLMError``.

        Args:
            request: Generation request including messages and parameters.

        Returns:
            Async iterator yielding generated text tokens.

        Raises:
            LLMError: On any provider error during streaming.
        """
        start = time.perf_counter()

        async def _inner() -> AsyncGenerator[str, None]:
            try:
                async for token in self._do_stream(request):
                    yield token
            except LLMError:
                raise
            except Exception as exc:
                raise self._wrap_exception(exc) from exc
            finally:
                observe_histogram(
                    _HISTOGRAM_NAME,
                    time.perf_counter() - start,
                    {
                        "provider": self.PROVIDER,
                        "model": self._settings.llm.model,
                        "operation": "stream",
                    },
                )

        return _inner()
