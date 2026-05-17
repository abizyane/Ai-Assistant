"""Document aggregate root — represents a single ingested source file."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Represents a source document (PDF, Markdown, HTML) ingested into the system."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_path: str = Field(description="Original file path or URL")
    content_hash: str = Field(description="SHA-256 hash of raw content for deduplication")
    language: str = Field(default="en", description="ISO-639-1 language code")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
