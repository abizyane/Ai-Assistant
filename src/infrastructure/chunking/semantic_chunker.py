"""Semantic Chunker adapter — layout-aware splitting that respects document sections."""

from __future__ import annotations

import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import Settings
from src.domain.entities.chunk import Chunk
from src.domain.ports.dto import RawDocument
from src.shared.errors import ChunkerError
from src.shared.metrics import observe_histogram
from src.shared.tracing import traced

__all__ = ["SemanticChunker"]

_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_CHUNK_OVERLAP = 100


class SemanticChunker:
    """Layout-aware document chunker implementing the ChunkerPort contract.

    Splits raw documents by section headings first (read from
    ``metadata["section_headings"]``), then applies
    ``RecursiveCharacterTextSplitter`` within each section so that no chunk
    ever exceeds ``chunk_size + chunk_overlap`` characters.

    Args:
        settings: Application settings; chunking sub-settings are read from
            ``settings.chunking.chunk_size`` and
            ``settings.chunking.chunk_overlap``.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise the chunker from application settings.

        Args:
            settings: Root application settings object.
        """
        self._chunk_size: int = settings.chunking.chunk_size
        self._chunk_overlap: int = settings.chunking.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
        )

    @traced("chunker.chunk")
    async def chunk(self, documents: list[RawDocument]) -> list[Chunk]:
        """Split a list of raw documents into ordered Chunk entities.

        Args:
            documents: Raw documents to split.

        Returns:
            Flat list of Chunk entities with sequential positions per document.

        Raises:
            ChunkerError: If a document cannot be processed.
        """
        result: list[Chunk] = []
        for doc in documents:
            try:
                chunks = self._chunk_document(doc)
            except Exception as exc:
                raise ChunkerError(
                    f"Failed to chunk document '{doc.source_path}'", cause=exc
                ) from exc
            observe_histogram("chunking_chunks_per_document", float(len(chunks)))
            result.extend(chunks)
        return result

    def _chunk_document(self, doc: RawDocument) -> list[Chunk]:
        """Split a single document respecting its section structure.

        Args:
            doc: Raw document to split.

        Returns:
            List of Chunk entities with zero-based sequential positions.
        """
        document_id = uuid.uuid5(uuid.NAMESPACE_URL, doc.content_hash)
        section_headings: list[str] = doc.metadata.get("section_headings", [])
        page_number: int = int(doc.metadata.get("page_number", 1))

        sections = self._split_by_sections(doc.content, section_headings)

        chunks: list[Chunk] = []
        position = 0

        for section_heading, section_text in sections:
            if not section_text.strip():
                continue
            sub_texts = self._splitter.split_text(section_text)
            for chunk_index, text in enumerate(sub_texts):
                if not text.strip():
                    continue
                metadata: dict[str, Any] = {
                    "source_path": doc.source_path,
                    "page_number": page_number,
                    "section_heading": section_heading,
                    "chunk_index": chunk_index,
                }
                chunks.append(
                    Chunk(
                        id=uuid.uuid4(),
                        document_id=document_id,
                        content=text,
                        position=position,
                        token_count=len(text.split()),
                        embedding=None,
                        metadata=metadata,
                    )
                )
                position += 1

        return chunks

    def _split_by_sections(
        self,
        content: str,
        section_headings: list[str],
    ) -> list[tuple[str | None, str]]:
        """Partition *content* into (heading, text) pairs using *section_headings*.

        The method searches for each heading as a literal substring.  Headings
        are sorted by their first occurrence so interleaved or out-of-order
        heading lists are handled correctly.  Text that appears before the
        first recognised heading is emitted with ``None`` as the heading.

        Args:
            content: Full document text.
            section_headings: Ordered list of section heading strings.

        Returns:
            List of ``(heading, text)`` pairs in document order.  Returns
            ``[(None, content)]`` when no headings are found in *content*.
        """
        if not section_headings:
            return [(None, content)]

        positions: list[tuple[int, str]] = []
        for heading in section_headings:
            idx = content.find(heading)
            if idx != -1:
                positions.append((idx, heading))

        if not positions:
            return [(None, content)]

        positions.sort(key=lambda x: x[0])

        sections: list[tuple[str | None, str]] = []

        first_pos = positions[0][0]
        if first_pos > 0:
            pre_text = content[:first_pos].strip()
            if pre_text:
                sections.append((None, pre_text))

        for i, (pos, heading) in enumerate(positions):
            next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(content)
            section_text = content[pos + len(heading) : next_pos].strip()
            if section_text:
                sections.append((heading, section_text))

        return sections if sections else [(None, content)]
