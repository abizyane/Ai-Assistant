"""Shared exception hierarchy for the RAG pipeline."""

from __future__ import annotations

__all__ = ["ChunkerError", "LoaderError", "RAGError"]


class RAGError(Exception):
    """Base exception for all RAG system errors.

    Attributes:
        message: Human-readable description of the error.
        cause: Underlying exception that caused this error, if any.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the error.
            cause: Underlying exception that triggered this error.
        """
        super().__init__(message)
        self.message = message
        self.cause = cause


class LoaderError(RAGError):
    """Raised when a document loader fails to load or process a file.

    Attributes:
        message: Human-readable description of the error.
        cause: Underlying exception that caused this error, if any.
    """


class ChunkerError(RAGError):
    """Raised when a document chunker fails to split a document.

    Attributes:
        message: Human-readable description of the error.
        cause: Underlying exception that caused this error, if any.
    """
