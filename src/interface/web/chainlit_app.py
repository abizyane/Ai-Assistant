"""Chainlit chat interface for the RAG assistant.

Streams answers from the FastAPI backend via SSE, renders citation cards,
and supports English / French / Arabic via a chat-settings language picker.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import chainlit as cl
import httpx

from src.interface.web.api_client import RAGAPIClient
from src.interface.web.i18n import SUPPORTED_LANGUAGES, load_translations

_RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8000")


def _make_client() -> RAGAPIClient:
    """Return a :class:`RAGAPIClient` pointed at :data:`_RAG_API_URL`."""
    return RAGAPIClient(base_url=_RAG_API_URL)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise per-session state and present the language-picker widget."""
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("language", "en")

    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="language",
                label="Language / Langue / اللغة",
                values=list(SUPPORTED_LANGUAGES),
                initial_value="en",
            )
        ]
    ).send()
    cl.user_session.set("language", settings.get("language", "en"))


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    """Persist the language preference when the user updates chat settings."""
    cl.user_session.set("language", settings.get("language", "en"))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream a response and render citation cards for *message*."""
    session_id: str = cl.user_session.get("session_id") or str(uuid.uuid4())
    language: str = cl.user_session.get("language") or "en"
    t: dict[str, Any] = load_translations(language)

    client = _make_client()
    msg = cl.Message(content="")
    await msg.send()

    citations: list[dict[str, Any]] = []

    try:
        async for event in client.chat_stream_async(
            query=message.content,
            session_id=session_id,
            language=language,
        ):
            if "token" in event:
                await msg.stream_token(event["token"])
            elif event.get("done"):
                citations = event.get("citations", [])
    except httpx.HTTPError as exc:
        await msg.update()
        error_label: str = t.get("error_network", "Network error: {error}").format(error=str(exc))
        await cl.Message(content=f"⚠️ {error_label}").send()
        return

    await msg.update()

    if citations:
        lines: list[str] = [t.get("citations_header", "**Sources**")]
        for cite in citations:
            source: str = cite.get("source", "")
            page: int | None = cite.get("page")
            excerpt: str = cite.get("excerpt", "")
            if page is not None:
                lines.append(f"- **{source}** (p. {page}): {excerpt}")
            else:
                lines.append(f"- **{source}**: {excerpt}")
        await cl.Message(content="\n".join(lines), parent_id=msg.id).send()
