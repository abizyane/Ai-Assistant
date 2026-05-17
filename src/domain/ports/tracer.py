"""Tracer port — contract for distributed tracing and observability adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.domain.ports.dto import TraceContext


@runtime_checkable
class TracerPort(Protocol):
    """Protocol for distributed tracing adapters used by the DI container."""

    def start_span(
        self,
        name: str,
        input: dict[str, Any] | None = None,  # noqa: A002
    ) -> TraceContext:
        """Start a new trace span and return its context."""
        ...

    def end_span(
        self,
        ctx: TraceContext,
        output: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Close the given trace span, recording output or error details."""
        ...

    def flush(self) -> None:
        """Flush all buffered trace data to the tracing backend."""
        ...
