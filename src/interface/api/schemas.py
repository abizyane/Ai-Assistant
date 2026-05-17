"""Pydantic v2 request/response schemas for the FastAPI REST API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChatRequest",
    "CitationOut",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "MessageOut",
    "SessionResponse",
    "SyncChatResponse",
]


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, description="User question text")
    session_id: str | None = Field(default=None, description="Existing session ULID (optional)")
    language: str | None = Field(default=None, description="BCP-47 language hint (optional)")


class CitationOut(BaseModel):
    """Serialisable citation for API responses."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="ID of the referenced chunk")
    source: str = Field(description="Human-readable source path")
    page: int | None = Field(default=None, description="Optional page number")
    marker: str = Field(description="Literal [chunk_id] marker emitted by LLM")


class SyncChatResponse(BaseModel):
    """Synchronous chat response — full answer with citations and token counts."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Generated answer text")
    citations: list[CitationOut] = Field(default_factory=list)
    language: str = Field(description="BCP-47 language code of the answer")
    tokens_in: int = Field(ge=0, description="LLM input token count")
    tokens_out: int = Field(ge=0, description="LLM output token count")
    session_id: str = Field(description="Session ULID used for this turn")


class IngestRequest(BaseModel):
    """Document ingestion request payload."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(min_length=1, description="Filesystem path to ingest (file or directory)")
    language_hint: str | None = Field(default=None, description="Override language detection")


class IngestResponse(BaseModel):
    """Document ingestion summary response."""

    model_config = ConfigDict(frozen=True)

    files: int = Field(ge=0, description="Number of files processed")
    chunks: int = Field(ge=0, description="Total chunks created")
    document_ids: list[str] = Field(
        default_factory=list, description="IDs of ingested documents (best-effort)"
    )


class MessageOut(BaseModel):
    """A single message in a session history response."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(description="Message role: user | assistant | system")
    content: str = Field(description="Message text content")


class SessionResponse(BaseModel):
    """Session detail response with message history."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(description="26-char ULID session identifier")
    messages: list[MessageOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """API health-check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="Overall status: ok | degraded | error")
    version: str = Field(description="Application version from pyproject.toml")
    db: str = Field(description="Database connectivity: ok | fail")
    langfuse: str = Field(description="Langfuse connectivity: ok | disabled | fail")
