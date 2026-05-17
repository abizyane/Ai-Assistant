"""Score value object — a typed numeric relevance or evaluation score."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ScoreKind(StrEnum):
    """Kind of score being represented."""

    DENSE = "dense"
    SPARSE = "sparse"
    RERANK = "rerank"
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"


class Score(BaseModel):
    """An immutable typed numeric score in [0, 1]."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0, description="Score in [0, 1]")
    kind: ScoreKind = Field(description="Type of score")
