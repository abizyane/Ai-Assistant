"""LLM adapter exception hierarchy."""

from __future__ import annotations

__all__ = ["LLMError", "LLMProviderError", "LLMRateLimitError", "LLMTimeoutError"]


class LLMError(Exception):
    """Base exception for all LLM adapter errors.

    All exceptions raised by LLM adapters wrap the underlying provider
    exception and attach metadata (provider name, original exception) to
    enable structured error handling at the application layer.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        original: Exception | None = None,
    ) -> None:
        """Initialize LLMError.

        Args:
            message: Human-readable error description.
            provider: LLM provider name (e.g. ``"gemini"``).
            original: Original provider exception that caused this error.
        """
        super().__init__(message)
        self.provider = provider
        self.original = original


class LLMProviderError(LLMError):
    """Raised when the LLM provider returns a server-side error (HTTP 5xx)."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate limit is exceeded (HTTP 429)."""


class LLMTimeoutError(LLMError):
    """Raised when a request to the LLM provider times out."""
