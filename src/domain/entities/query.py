"""Query entity — a user query entering the agent pipeline."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class QueryIntent(StrEnum):
    """Classified intent of a user query."""

    RAG = "rag"
    SMALL_TALK = "small_talk"
    OUT_OF_SCOPE = "out_of_scope"


class Query(BaseModel):
    """A user query entering the RAG agent pipeline."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID = Field(description="Parent session ID")
    text: str = Field(min_length=1, description="Raw user query text")
    language: str = Field(default="en", description="Detected ISO-639-1 language")
    intent: QueryIntent = Field(default=QueryIntent.RAG)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
