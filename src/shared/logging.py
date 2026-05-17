"""Structured JSON logging via structlog with contextvars support."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

from src.shared.tracing import current_session_id, current_trace_id

# ContextVars for request-scoped fields (trace/session propagation)
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")


def _inject_otel_context(
    logger: Any,  # noqa: ANN401
    method: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    if "trace_id" not in event_dict:
        tid = current_trace_id.get()
        if tid is not None:
            event_dict["trace_id"] = tid
    if "session_id" not in event_dict:
        sid = current_session_id.get()
        if sid is not None:
            event_dict["session_id"] = sid
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON renderer and contextvars injection."""
    # shared processors (used by both structlog and stdlib bridge)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_otel_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name."""
    return structlog.get_logger(name)  # type: ignore[no-any-return] — structlog.get_logger returns BoundLogger, mypy infers Any


def bind_context(*, trace_id: str = "", session_id: str = "") -> None:
    """Bind trace/session IDs to contextvars for log correlation."""
    structlog.contextvars.bind_contextvars(
        trace_id=trace_id or _trace_id.get(),
        session_id=session_id or _session_id.get(),
    )


def clear_context() -> None:
    """Clear all bound contextvars (call in tests between cases)."""
    structlog.contextvars.clear_contextvars()
