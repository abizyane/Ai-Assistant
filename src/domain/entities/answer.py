"""Answer entities and DTOs — domain answers and use-case output shapes."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.domain.entities.citation import Citation


class AnswerCitation(BaseModel):
    """A single citation extracted from a generated answer.

    Distinct from ``src.domain.entities.citation.Citation`` (which is the
    persistence-layer entity keyed by UUIDs). ``AnswerCitation`` is the
    DTO returned to callers: it carries the raw chunk identifier as a
    string, the human-readable source path, an optional page number,
    and the inline marker the LLM emitted (e.g. ``[ref-1]``).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1, description="Identifier of the cited chunk")
    source: str = Field(min_length=1, description="Human-readable source path")
    page: int | None = Field(default=None, description="1-indexed page number, if known")
    marker: str = Field(min_length=1, description="Inline marker as it appears in the answer")


class AnswerWithCitations(BaseModel):
    """Grounded answer returned by the generate-answer use case."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Generated answer text")
    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description="Citations grounded in retrieved chunks",
    )
    language: str = Field(min_length=2, description="BCP-47 language code")
    tokens_in: int = Field(ge=0, description="Input tokens consumed by the LLM")
    tokens_out: int = Field(ge=0, description="Output tokens produced by the LLM")


class Answer(BaseModel):
    """Domain entity representing an answer produced for a user query."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query_id: uuid.UUID = Field(description="ID of the query this answers")
    text: str = Field(min_length=1, description="Generated answer text")
    confidence: float = Field(ge=0.0, le=1.0, description="Answer confidence score 0-1")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations grounding the answer",
    )
