"""ASGI middleware: request-ID structlog binding, session contextvar, duration histogram."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from ulid import ULID

from src.shared.metrics import observe_histogram
from src.shared.tracing import current_session_id

__all__ = ["RequestLoggingMiddleware"]

log = structlog.get_logger(__name__)

_REQUEST_DURATION_METRIC = "http_request_duration_seconds"
_REQUEST_ID_HEADER = "X-Request-Id"
_SESSION_ID_HEADER = "X-Session-Id"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Per-request structlog binding, session contextvar injection, and duration histogram.

    For every incoming HTTP request this middleware:

    1. Reads or generates a ULID request-ID from ``X-Request-Id``.
    2. Reads or generates a ULID session-ID from ``X-Session-Id``.
    3. Binds both to structlog contextvars so every downstream log line is
       correlated.
    4. Sets the ``current_session_id`` contextvar (from
       ``src.shared.tracing``) so OTel spans and LangGraph nodes pick it up.
    5. Measures wall-clock duration and records
       ``http_request_duration_seconds{method, path, status}`` histogram.
    6. Echoes the session-ID back in the ``X-Session-Id`` response header.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request: bind contextvars, call next handler, record metrics."""
        request_id = request.headers.get(_REQUEST_ID_HEADER) or str(ULID())
        session_id = request.headers.get(_SESSION_ID_HEADER) or str(ULID())

        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            session_id=session_id,
        )
        token = current_session_id.set(session_id)

        start = time.perf_counter()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        finally:
            duration = time.perf_counter() - start
            observe_histogram(
                _REQUEST_DURATION_METRIC,
                duration,
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": str(status_code),
                },
            )
            current_session_id.reset(token)
            structlog.contextvars.unbind_contextvars("request_id", "session_id")

        response.headers[_SESSION_ID_HEADER] = session_id
        return response
