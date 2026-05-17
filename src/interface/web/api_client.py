"""Async httpx wrapper for the RAG FastAPI backend.

All communication with the backend goes through this module.
No imports from ``src.infrastructure`` or ``src.application`` are permitted here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

__all__ = ["RAGAPIClient"]

log = logging.getLogger(__name__)


class RAGAPIClient:
    """Thin async wrapper around the RAG REST API.

    Args:
        base_url: Base URL of the FastAPI backend, e.g. ``http://localhost:8000``.
        langfuse_base_url: Base URL of the Langfuse UI for trace deep-links.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        langfuse_base_url: str = "http://localhost:3000",
    ) -> None:
        """Initialise the client with API and Langfuse base URLs."""
        self._base_url = base_url.rstrip("/")
        self._langfuse_base_url = langfuse_base_url.rstrip("/")

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def langfuse_trace_url(self, trace_id: str) -> str:
        """Return Langfuse deep-link URL for a trace.

        Args:
            trace_id: Trace identifier returned by the API.

        Returns:
            Fully-qualified Langfuse trace URL string.
        """
        return f"{self._langfuse_base_url}/trace/{trace_id}"

    # ------------------------------------------------------------------
    # Async API methods
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """GET /health → health status dict.

        Returns:
            Parsed JSON response from the health endpoint.

        Raises:
            httpx.HTTPError: On HTTP error or connection failure.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self._base_url}/health")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def chat_sync(
        self,
        query: str,
        session_id: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """POST /chat/sync → full answer JSON.

        Args:
            query: The user question.
            session_id: Optional existing session ULID.
            language: Optional BCP-47 language hint.

        Returns:
            Parsed JSON dict with ``text``, ``citations``, ``session_id``, etc.

        Raises:
            httpx.HTTPError: On HTTP error or connection failure.
        """
        payload: dict[str, Any] = {"query": query}
        if session_id:
            payload["session_id"] = session_id
        if language:
            payload["language"] = language
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}/chat/sync", json=payload)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def chat_stream_async(
        self,
        query: str,
        session_id: str | None = None,
        language: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """POST /chat (SSE) → async generator of event dicts.

        Args:
            query: The user question.
            session_id: Optional existing session ULID.
            language: Optional BCP-47 language hint.

        Yields:
            Parsed event dicts from the SSE stream.

        Raises:
            httpx.HTTPError: On HTTP error or connection failure.
        """
        payload: dict[str, Any] = {"query": query}
        if session_id:
            payload["session_id"] = session_id
        if language:
            payload["language"] = language
        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream("POST", f"{self._base_url}/chat", json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        log.warning("unparseable SSE line: %s", line)

    def chat_stream_sync(
        self,
        query: str,
        session_id: str | None = None,
        language: str | None = None,
        citations_out: list[Any] | None = None,
    ) -> Iterator[str]:
        """Synchronous token generator for Streamlit.

        Runs :meth:`chat_stream_async` in a daemon thread so Streamlit's
        synchronous rendering loop can consume tokens one at a time.  When
        the stream completes, citation dicts from the final ``done`` event
        are appended to *citations_out*.

        Args:
            query: The user question.
            session_id: Optional existing session ULID.
            language: Optional BCP-47 language hint.
            citations_out: Mutable list; citation dicts appended after streaming.

        Yields:
            Token strings in arrival order.

        Raises:
            Exception: Any exception raised by the async producer is re-raised
                in the calling thread after the generator exhausts.
        """
        token_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        exc_holder: list[BaseException] = []

        async def _producer() -> None:
            try:
                async for event in self.chat_stream_async(query, session_id, language):
                    if "token" in event:
                        token_queue.put(("token", event["token"]))
                    elif event.get("done"):
                        token_queue.put(("done", event.get("citations", [])))
                        return
                # Stream ended without an explicit done event
                token_queue.put(("done", []))
            except Exception as exc:
                exc_holder.append(exc)
                token_queue.put(("done", []))

        def _run_loop() -> None:
            asyncio.run(_producer())

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()

        while True:
            kind, value = token_queue.get(timeout=120.0)
            if kind == "done":
                if citations_out is not None:
                    citations_out.extend(value)
                break
            yield str(value)

        thread.join(timeout=5.0)
        if exc_holder:
            raise exc_holder[0]

    async def ingest(
        self,
        path: str,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        """POST /ingest → ingestion summary dict.

        Args:
            path: Filesystem path to ingest (file or directory).
            language_hint: Optional language code override for detection.

        Returns:
            Parsed JSON dict with ``files``, ``chunks``, ``document_ids``.

        Raises:
            httpx.HTTPError: On HTTP error or connection failure.
        """
        payload: dict[str, Any] = {"path": path}
        if language_hint:
            payload["language_hint"] = language_hint
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self._base_url}/ingest", json=payload)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """GET /sessions/{session_id} → session history dict.

        Args:
            session_id: 26-char ULID session identifier.

        Returns:
            Parsed JSON dict with ``session_id`` and ``messages`` list.

        Raises:
            httpx.HTTPError: On HTTP error or connection failure.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self._base_url}/sessions/{session_id}")
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
