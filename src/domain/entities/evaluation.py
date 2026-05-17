"""Evaluation report entity — Ragas-based RAG quality snapshot."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationReport(BaseModel):
    """Immutable result of a Ragas evaluation run.

    Attributes:
        timestamp: UTC instant when the evaluation completed.
        dataset_path: Path to the JSONL dataset used.
        sample_size: Number of rows evaluated (after optional sampling).
        per_question: Per-row evaluation data including query, ground_truth,
            prediction, contexts, and per-metric scores.
        aggregate: Mean per metric over all evaluated rows.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(description="UTC instant when the evaluation completed")
    dataset_path: str = Field(description="Path to the JSONL evaluation dataset")
    sample_size: int = Field(ge=1, description="Number of rows evaluated")
    per_question: list[dict[str, Any]] = Field(
        description="Per-row data: query, ground_truth, prediction, contexts, metric scores",
    )
    aggregate: dict[str, float] = Field(
        description="Mean metric scores: faithfulness, answer_relevancy, "
        "context_precision, context_recall, answer_correctness",
    )

    def to_json(self) -> str:
        """Return this report serialised as a JSON string.

        Returns:
            A UTF-8 JSON string with ISO-8601 datetime encoding.
        """
        return self.model_dump_json()
