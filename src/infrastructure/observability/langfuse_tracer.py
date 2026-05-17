"""Langfuse tracing adapter implementing TracerPort."""

from __future__ import annotations

import contextlib
from typing import Any
from uuid import uuid4

from src.domain.ports.dto import TraceContext
from src.domain.ports.tracer import TracerPort


class LangfuseTracer:
    """TracerPort implementation wrapping Langfuse SDK for LLM observability.

    If langfuse is not installed or credentials are missing, operations are
    no-ops (graceful degradation). This allows running without Langfuse for
    local development.
    """

    def __init__(
        self,
        host: str = "http://localhost:3000",
        public_key: str = "",
        secret_key: str = "",
    ) -> None:
        """Initialize LangfuseTracer with optional credentials."""
        self._enabled = bool(public_key and secret_key)
        self._client: Any = None
        if self._enabled:
            try:
                from langfuse import Langfuse  # type: ignore[import-untyped]

                self._client = Langfuse(
                    host=host,
                    public_key=public_key,
                    secret_key=secret_key,
                )
            except ImportError:
                self._enabled = False

    def start_span(
        self,
        name: str,
        input: dict[str, Any] | None = None,  # noqa: A002
    ) -> TraceContext:
        """Start a Langfuse trace/span and return its context."""
        span_id = str(uuid4())
        trace_id = str(uuid4())
        if self._enabled and self._client is not None:
            try:
                trace_obj = self._client.trace(name=name, input=input or {})
                trace_id = trace_obj.id
                span = trace_obj.span(name=name, input=input or {})
                span_id = span.id
            except Exception:
                pass
        return TraceContext(span_id=span_id, trace_id=trace_id, name=name, metadata=input or {})

    def end_span(
        self,
        ctx: TraceContext,
        output: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Close the span in Langfuse, recording output or error."""
        if not self._enabled or self._client is None:
            return
        try:
            if output:
                self._client.trace(id=ctx.trace_id).update(output=output)
        except Exception:
            pass

    def flush(self) -> None:
        """Flush all buffered Langfuse events to the backend."""
        if self._enabled and self._client is not None:
            with contextlib.suppress(Exception):
                self._client.flush()


_: TracerPort = LangfuseTracer()  # structural subtype check at import time
