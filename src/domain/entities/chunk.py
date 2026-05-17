"""Chunk entity — a semantic segment of a Document."""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A text chunk extracted from a Document, ready for embedding."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document_id: uuid.UUID = Field(description="Parent document ID")
    content: str = Field(min_length=1, description="Raw text content of this chunk")
    position: int = Field(ge=0, description="Zero-based position index within the document")
    token_count: int = Field(ge=0, description="Estimated token count")
    embedding: list[float] | None = Field(default=None, description="Dense embedding vector")
    metadata: dict[str, Any] = Field(default_factory=dict)
