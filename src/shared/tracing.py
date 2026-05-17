"""OpenTelemetry SDK bootstrap with OTLP exporter targeting Langfuse."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import AsyncGenerator, Callable
from contextvars import ContextVar
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.status import Status, StatusCode

__all__ = [
    "configure_tracing",
    "current_session_id",
    "current_trace_id",
    "get_tracer",
    "reset_tracing",
    "traced",
]

# Module-level contextvars for OTel trace/session propagation across awaits
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)

_initialized = False

_F = TypeVar("_F", bound=Callable[..., Any])


def configure_tracing(
    service_name: str = "rag-assistant",
    otlp_endpoint: str | None = None,
    *,
    enabled: bool = True,
) -> None:
    """Bootstrap OTel SDK. Call once at startup. Idempotent."""
    global _initialized
    if _initialized:
        return
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    if enabled and otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str) -> trace.Tracer:
    """Return an OTel tracer for the given module/component name."""
    return trace.get_tracer(name)


def reset_tracing() -> None:
    """Reset for testing — allows re-initialization."""
    global _initialized
    _initialized = False
    trace.set_tracer_provider(trace.ProxyTracerProvider())


def traced(span_name: str, **default_attrs: Any) -> Callable[[_F], _F]:  # noqa: ANN401
    """Decorator that wraps an async function in an OTel span.

    Supports both regular coroutines and async generator functions. For regular
    coroutines, auto-extracts ``gen_ai.usage.input_tokens`` /
    ``gen_ai.usage.output_tokens`` from the result when the result object
    carries those attributes (e.g. ``GenerationResult``). Records exceptions
    via ``span.record_exception`` and marks the span status ``ERROR`` before
    re-raising.  Propagates ``current_trace_id`` / ``current_session_id``
    contextvars as span attributes.

    Args:
        span_name: Name for the OTel span.
        **default_attrs: Default span attributes set at span creation.

    Returns:
        Decorator that preserves the wrapped function's call signature.
    """

    def decorator(func: _F) -> _F:
        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def _asyncgen_wrapper(
                *args: Any, **kwargs: Any  # noqa: ANN401
            ) -> AsyncGenerator[Any, None]:
                tracer = get_tracer(func.__module__ or __name__)
                attrs = dict(default_attrs) if default_attrs else None
                with tracer.start_as_current_span(
                    span_name,
                    attributes=attrs,
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    _apply_context_attrs(span)
                    try:
                        async for item in func(*args, **kwargs):
                            yield item
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR))
                        raise

            return _asyncgen_wrapper  # type: ignore[return-value] — wrapper preserves callable signature via functools.wraps

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                tracer = get_tracer(func.__module__ or __name__)
                attrs = dict(default_attrs) if default_attrs else None
                with tracer.start_as_current_span(
                    span_name,
                    attributes=attrs,
                    record_exception=False,
                    set_status_on_exception=False,
                ) as span:
                    _apply_context_attrs(span)
                    try:
                        result: Any = await func(*args, **kwargs)
                        _extract_gen_ai_usage(span, result)
                        return result
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR))
                        raise

            return _async_wrapper  # type: ignore[return-value] — wrapper preserves callable signature via functools.wraps

        # Non-async: return unchanged (no-op wrapping)
        return func

    return decorator


def _apply_context_attrs(span: trace.Span) -> None:
    tid = current_trace_id.get()
    sid = current_session_id.get()
    if tid is not None:
        span.set_attribute("trace.id", tid)
    if sid is not None:
        span.set_attribute("session.id", sid)


def _extract_gen_ai_usage(span: trace.Span, result: Any) -> None:  # noqa: ANN401
    if hasattr(result, "input_tokens") and hasattr(result, "output_tokens"):
        span.set_attribute("gen_ai.usage.input_tokens", result.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", result.output_tokens)
