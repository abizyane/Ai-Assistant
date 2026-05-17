"""LLM port — contract for language model generation adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from src.domain.ports.dto import GenerationRequest, GenerationResult


@runtime_checkable
class LLMPort(Protocol):
    """Protocol for language model generation adapters."""

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a complete text response for the given request."""
        ...

    async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Stream generated text tokens for the given request."""
        ...
