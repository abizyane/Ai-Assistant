"""Citation entity — a source chunk referenced in an Answer."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A reference to a source chunk supporting an Answer."""

    model_config = ConfigDict(frozen=True)

    chunk_id: uuid.UUID = Field(description="ID of the referenced Chunk")
    document_id: uuid.UUID = Field(description="ID of the parent Document")
    source_path: str = Field(description="Human-readable source path")
    snippet: str = Field(min_length=1, description="Quoted text from the chunk")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score in [0, 1]")
