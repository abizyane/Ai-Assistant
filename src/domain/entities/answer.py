"""Answer entity — the agent's response to a Query."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from src.domain.entities.citation import Citation


class Answer(BaseModel):
    """The agent's generated response, with citations and confidence score."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query_id: uuid.UUID = Field(description="ID of the originating Query")
    text: str = Field(min_length=1, description="Generated answer text")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence in [0, 1]")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    regeneration_count: int = Field(default=0, ge=0, description="How many times regenerated")
