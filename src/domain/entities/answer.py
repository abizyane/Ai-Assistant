"""Answer DTOs returned by the generate-answer use case."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
