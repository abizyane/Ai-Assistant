"""Session and Message entities — conversation state."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    """Speaker role in a conversation message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single turn in a conversation session."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID = Field(description="Parent session ID")
    role: MessageRole = Field(description="Speaker role")
    content: str = Field(min_length=1, description="Message text content")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Session(BaseModel):
    """A user conversation session."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: str | None = Field(default=None, description="Optional user identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = Field(default_factory=lambda: datetime.now(UTC))
