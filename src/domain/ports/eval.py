"""Eval port — contract for RAG evaluation framework adapters."""

from __future__ import annotations

from typing import Protocol

from src.domain.ports.dto import EvalRequest, EvalResult


class EvalPort(Protocol):
    """Protocol for RAG evaluation framework adapters."""

    async def evaluate(self, request: EvalRequest) -> EvalResult:
        """Run an evaluation pass and return aggregated quality metrics."""
        ...
