"""Unit tests for SemanticChunker — layout-aware document chunking adapter."""

from __future__ import annotations

import hashlib

import pytest

from src.config.settings import Settings
from src.domain.ports.dto import RawDocument
from src.infrastructure.chunking.semantic_chunker import SemanticChunker


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _doc(
    content: str,
    source_path: str = "test.pdf",
    metadata: dict | None = None,
) -> RawDocument:
    return RawDocument(
        source_path=source_path,
        content=content,
        language="en",
        content_hash=_hash(content),
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_section_metadata_preserved() -> None:
    """Chunks carry the correct section_heading from parent document metadata."""
    content = "Intro\nThis is the intro.\n\nBody\nThis is the body.\n\nConclusion\nThis is the conclusion."
    doc = _doc(
        content=content,
        metadata={"section_headings": ["Intro", "Body", "Conclusion"]},
    )
    chunker = SemanticChunker(Settings(llm={"api_key": "test"}))  # type: ignore[arg-type]
    chunks = await chunker.chunk([doc])

    headings = {c.metadata["section_heading"] for c in chunks}
    assert "Intro" in headings or "Body" in headings or "Conclusion" in headings


@pytest.mark.asyncio
async def test_chunk_size_bounds() -> None:
    """All chunks must not exceed chunk_size + chunk_overlap characters."""
    chunk_size = 50
    chunk_overlap = 10
    long_text = "word " * 500
    doc = _doc(content=long_text)

    s = Settings(llm={"api_key": "test"})  # type: ignore[arg-type]
    object.__setattr__(s.chunking, "chunk_size", chunk_size)
    object.__setattr__(s.chunking, "chunk_overlap", chunk_overlap)

    chunker = SemanticChunker(s)
    chunks = await chunker.chunk([doc])

    assert chunks, "Expected at least one chunk from a long document"
    for chunk in chunks:
        assert len(chunk.content) <= chunk_size + chunk_overlap, (
            f"Chunk too long: {len(chunk.content)} > {chunk_size + chunk_overlap}"
        )


@pytest.mark.asyncio
async def test_chunk_position_sequential() -> None:
    """Chunk positions are zero-based and strictly sequential."""
    long_text = "sentence " * 300
    doc = _doc(content=long_text)

    s = Settings(llm={"api_key": "test"})  # type: ignore[arg-type]
    object.__setattr__(s.chunking, "chunk_size", 100)
    object.__setattr__(s.chunking, "chunk_overlap", 10)

    chunker = SemanticChunker(s)
    chunks = await chunker.chunk([doc])

    assert len(chunks) > 1, "Expected multiple chunks from a long document"
    positions = [c.position for c in chunks]
    assert positions == list(range(len(chunks)))


@pytest.mark.asyncio
async def test_empty_doc_returns_empty() -> None:
    """An empty document yields an empty chunk list."""
    doc = _doc(content="")
    chunker = SemanticChunker(Settings(llm={"api_key": "test"}))  # type: ignore[arg-type]

    chunks = await chunker.chunk([doc])
    assert chunks == []


@pytest.mark.asyncio
async def test_metadata_inherited() -> None:
    """Every chunk carries the source_path from its parent document."""
    source = "my/document.pdf"
    doc = _doc(content="Some text content for testing metadata inheritance.", source_path=source)
    chunker = SemanticChunker(Settings(llm={"api_key": "test"}))  # type: ignore[arg-type]

    chunks = await chunker.chunk([doc])

    assert chunks, "Expected at least one chunk"
    for chunk in chunks:
        assert chunk.metadata["source_path"] == source
