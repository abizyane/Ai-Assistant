"""Embedding value object — immutable dense vector with metadata."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Embedding(BaseModel):
    """An immutable dense embedding vector with dimension validation."""

    model_config = ConfigDict(frozen=True)

    vector: tuple[float, ...] = Field(min_length=1, description="Dense vector components")
    dimension: int = Field(gt=0, description="Expected vector dimension")
    model: str = Field(min_length=1, description="Embedding model identifier")

    @model_validator(mode="after")
    def _validate_dimension(self) -> Embedding:
        """Validate vector length matches declared dimension."""
        if len(self.vector) != self.dimension:
            msg = f"vector length {len(self.vector)} != dimension {self.dimension}"
            raise ValueError(msg)
        return self
