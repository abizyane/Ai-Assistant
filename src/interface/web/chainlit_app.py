"""Chainlit chat interface for the RAG assistant.

Streams answers from the FastAPI backend via SSE, renders citation cards,
and supports English / French / Arabic via chat profiles and a language-picker
widget. Conversation history persists via data_persistence in config.toml.
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

# Maps chat profile display name → BCP-47 language code.
_PROFILE_LANG: dict[str, str] = {
    "English": "en",
    "Français": "fr",
    "العربية": "ar",
}

# Per-language welcome messages shown at the start of every session.
_WELCOME: dict[str, str] = {
    "en": (
        "## 1337 AI Assistant\n\n"
        "Ask me anything about **1337 Coding School** \u2014 admissions, the selection "
        "process (Piscine), campus life, curriculum, or the peer-to-peer pedagogy.\n\n"
        "I answer in **English**, **French**, and **العربية** \u2014 pick your language "
        "via the profile selector before starting, or switch it in the settings below.\n\n"
        "> Grounded in the official 1337 knowledge base via a multilingual RAG pipeline."
    ),
    "fr": (
        "## Assistant 1337\n\n"
        "Posez vos questions sur **l'\u00e9cole 1337** \u2014 admissions, la Piscine, la vie "
        "sur campus, les cursus ou la p\u00e9dagogie entre pairs.\n\n"
        "Je r\u00e9ponds en **anglais**, **fran\u00e7ais** et **العربية** \u2014 choisissez votre "
        "langue dans le s\u00e9lecteur de profil ou modifiez-la dans les param\u00e8tres ci-dessous.\n\n"
        "> Ancr\u00e9 dans la base de connaissances officielle de 1337 via un pipeline RAG multilingue."
    ),
    "ar": (
        "## مساعد 1337 الذكي\n\n"
        "اسألني أي شيء عن **مدرسة 1337** \u2014 القبول، Piscine، الحياة الجامعية، "
        "المناهج أو التعلّم بين الأقران.\n\n"
        "أجيب بـ**الإنجليزية** و**الفرنسية** و**العربية** \u2014 اختر لغتك من الملف "
        "الشخصي أو غيّرها من الإعداد أدناه.\n\n"
        "> مبني على قاعدة المعرفة الرسمية لـ 1337 عبر خط أنابيب RAG متعدد اللغات."
    ),
}


def _make_client() -> RAGAPIClient:
    """Return a :class:`RAGAPIClient` pointed at :data:`_RAG_API_URL`."""
    return RAGAPIClient(base_url=_RAG_API_URL)


@cl.set_chat_profiles
async def chat_profiles() -> list[cl.ChatProfile]:
    """Define language-based chat profiles shown before a conversation starts."""
    return [
        cl.ChatProfile(
            name="English",
            markdown_description="Chat about 1337 Coding School in **English**.",
            icon="/public/avatar.svg",
        ),
        cl.ChatProfile(
            name="Fran\u00e7ais",
            markdown_description="Discutez de l'\u00e9cole 1337 en **fran\u00e7ais**.",
            icon="/public/avatar.svg",
        ),
        cl.ChatProfile(
            name="العربية",
            markdown_description="تحدث عن مدرسة 1337 بـ**العربية**.",
            icon="/public/avatar.svg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise per-session state and present the welcome message."""
    profile_name: str = cl.context.session.chat_profile or "English"
    language: str = _PROFILE_LANG.get(profile_name, "en")
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("language", language)

    # Language picker lets users override the profile choice mid-session.
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="language",
                label="Language / Langue / اللغة",
                values=list(SUPPORTED_LANGUAGES),
                initial_value=language,
            )
        ]
    ).send()
    cl.user_session.set("language", settings.get("language", language))

    welcome_text: str = _WELCOME.get(language, _WELCOME["en"])
    await cl.Message(content=welcome_text, author="1337 AI").send()

@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    """Persist the language preference when the user updates chat settings."""
    cl.user_session.set("language", settings.get("language", "en"))


@cl.on_chat_resume
async def on_chat_resume(thread: cl.types.ThreadDict) -> None:
    """Restore session state when resuming a conversation from the sidebar."""
    cl.user_session.set("session_id", thread.get("id", str(uuid.uuid4())))
    metadata: dict[str, Any] = thread.get("metadata") or {}
    cl.user_session.set("language", metadata.get("language", "en"))


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
        await cl.Message(content=f"[ERROR] {error_label}").send()
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
