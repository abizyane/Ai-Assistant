"""Docling-backed document loader — PDF, Markdown, HTML ingestion adapter."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from src.config.settings import Settings
from src.domain.ports.dto import RawDocument
from src.shared.errors import LoaderError
from src.shared.metrics import observe_histogram
from src.shared.tracing import traced

__all__ = ["DoclingLoader"]

_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".md", ".html"})

_log = logging.getLogger(__name__)


class DoclingLoader:
    """Document loader adapter backed by Docling for structure-preserving ingestion.

    Supports PDF, Markdown, and HTML files.  Accepts either a single file path
    or a directory; directories are recursed and all matching files are converted
    concurrently using ``asyncio.to_thread`` so the event loop is never blocked.

    Args:
        settings: Application settings (unused at runtime but required for DI
            consistency and potential future configuration).
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the loader and create a Docling ``DocumentConverter``.

        Args:
            settings: Application settings instance.

        Raises:
            ImportError: If docling is not installed.
        """
        from docling.document_converter import DocumentConverter

        self._settings = settings
        self._converter = DocumentConverter()

    @traced("loader.load")
    async def load(self, source_path: str | Path) -> list[RawDocument]:
        """Load documents from *source_path* (file or directory).

        For directories every ``.pdf``, ``.md``, and ``.html`` file is
        discovered recursively.  Files are converted concurrently; a failed
        individual file emits a warning and is skipped rather than aborting the
        entire batch.

        Args:
            source_path: Absolute or relative path to a file or directory.

        Returns:
            Ordered list of :class:`~src.domain.ports.dto.RawDocument` objects,
            one per successfully converted file.

        Raises:
            LoaderError: If *source_path* does not exist.
        """
        path = Path(source_path)
        if not path.exists():
            raise LoaderError(f"Path does not exist: {path}")

        file_paths = _collect_files(path)
        if not file_paths:
            _log.warning("No supported files found in %s", path)
            return []

        tasks = [self._load_one(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents: list[RawDocument] = []
        for fp, result in zip(file_paths, results, strict=False):
            if isinstance(result, BaseException):
                _log.warning("Skipping %s — conversion failed: %s", fp, result)
            else:
                documents.append(result)
        return documents

    def supports(self, source_path: str) -> bool:
        """Return ``True`` if the path points to a supported file or directory.

        Args:
            source_path: Path string to check.

        Returns:
            ``True`` for directories and for files with ``.pdf``, ``.md``, or
            ``.html`` suffix.
        """
        path = Path(source_path)
        if path.is_dir():
            return True
        return path.suffix.lower() in _SUPPORTED_SUFFIXES

    async def _load_one(self, file_path: Path) -> RawDocument:
        """Convert a single file and return a :class:`RawDocument`.

        The blocking Docling call is executed in a thread pool via
        :func:`asyncio.to_thread`.  Duration is recorded as a histogram metric
        labelled with the file format.

        Args:
            file_path: Path to the individual file.

        Returns:
            A populated :class:`~src.domain.ports.dto.RawDocument`.

        Raises:
            Exception: Re-raises any exception from Docling so the caller can
                decide whether to skip or abort.
        """
        fmt = file_path.suffix.lstrip(".").lower()
        start = time.perf_counter()
        result = await asyncio.to_thread(self._converter.convert, str(file_path))
        duration = time.perf_counter() - start
        observe_histogram(
            "document_load_duration_seconds",
            duration,
            {"format": fmt},
        )
        return _result_to_raw_document(result, file_path)


def _collect_files(path: Path) -> list[Path]:
    """Return all supported files under *path*.

    If *path* is a file its suffix is checked; if it is a directory the tree
    is walked recursively.

    Args:
        path: A file or directory path that is known to exist.

    Returns:
        List of :class:`~pathlib.Path` objects matching supported suffixes.
    """
    if path.is_file():
        return [path] if path.suffix.lower() in _SUPPORTED_SUFFIXES else []
    return [
        child
        for suffix in _SUPPORTED_SUFFIXES
        for child in path.rglob(f"*{suffix}")
        if child.is_file()
    ]


def _extract_headings(document: object) -> list[str]:
    """Extract section heading texts from a :class:`DoclingDocument`.

    Args:
        document: A ``DoclingDocument`` instance.

    Returns:
        Ordered list of heading strings, possibly empty.
    """
    try:
        from docling_core.types.doc import SectionHeaderItem  # type: ignore[attr-defined]

        return [item.orig for item in document.texts if isinstance(item, SectionHeaderItem)]  # type: ignore[attr-defined]
    except Exception:
        return []


def _result_to_raw_document(result: object, file_path: Path) -> RawDocument:
    """Convert a Docling :class:`ConversionResult` into a :class:`RawDocument`.

    Args:
        result: The ``ConversionResult`` returned by ``DocumentConverter.convert``.
        file_path: Original file path (used for ``source_path`` and format metadata).

    Returns:
        A populated :class:`~src.domain.ports.dto.RawDocument`.
    """
    doc = result.document  # type: ignore[attr-defined]
    text = doc.export_to_markdown()
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    page_count: int = len(doc.pages) if hasattr(doc, "pages") else 0
    has_tables: bool = bool(doc.tables) if hasattr(doc, "tables") else False
    headings: list[str] = _extract_headings(doc)
    fmt = file_path.suffix.lstrip(".").lower()

    metadata: dict[str, Any] = {
        "document_id": str(uuid.uuid4()),
        "file_format": fmt,
        "has_tables": has_tables,
        "language": "",
        "page_count": page_count,
        "section_headings": headings,
    }

    return RawDocument(
        source_path=str(file_path),
        content=text,
        language="",
        content_hash=content_hash,
        metadata=metadata,
    )
