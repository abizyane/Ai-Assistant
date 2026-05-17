"""Tests for domain entities."""
from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from src.domain.entities.chunk import Chunk
from src.domain.entities.citation import Citation
from src.domain.entities.document import Document
from src.domain.entities.query import Query, QueryIntent
from src.domain.entities.session import Message, MessageRole, Session


class TestDocument:
    def test_document_constructs_with_defaults(self) -> None:
        doc = Document(source_path="/data/file.pdf", content_hash="abc123")
        assert isinstance(doc.id, uuid.UUID)
        assert doc.language == "en"
        assert doc.metadata == {}
        assert doc.created_at is not None

    def test_document_frozen_mutation_raises(self) -> None:
        doc = Document(source_path="/data/file.pdf", content_hash="abc123")
        with pytest.raises(ValidationError):
            doc.language = "fr"  # type: ignore[misc]

    def test_document_custom_metadata(self) -> None:
        doc = Document(
            source_path="/data/file.pdf",
            content_hash="sha256:deadbeef",
            language="fr",
            metadata={"pages": 10},
        )
        assert doc.language == "fr"
        assert doc.metadata["pages"] == 10


class TestChunk:
    def test_chunk_requires_document_id(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(content="hello", position=0, token_count=1)  # type: ignore[call-arg]

    def test_chunk_constructs_with_document_id(self) -> None:
        doc_id = uuid.uuid4()
        chunk = Chunk(document_id=doc_id, content="hello world", position=0, token_count=2)
        assert chunk.document_id == doc_id
        assert chunk.embedding is None

    def test_chunk_empty_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(document_id=uuid.uuid4(), content="", position=0, token_count=0)

    def test_chunk_negative_position_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(document_id=uuid.uuid4(), content="x", position=-1, token_count=0)


class TestQuery:
    def test_query_default_intent_is_rag(self) -> None:
        q = Query(session_id=uuid.uuid4(), text="What is 1337?")
        assert q.intent == QueryIntent.RAG

    def test_query_explicit_intent(self) -> None:
        q = Query(
            session_id=uuid.uuid4(),
            text="Hey there!",
            intent=QueryIntent.SMALL_TALK,
        )
        assert q.intent == QueryIntent.SMALL_TALK

    def test_query_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            Query(session_id=uuid.uuid4(), text="")

    def test_query_frozen_mutation_raises(self) -> None:
        q = Query(session_id=uuid.uuid4(), text="hello")
        with pytest.raises(ValidationError):
            q.text = "changed"  # type: ignore[misc]


class TestCitation:
    def test_citation_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            Citation(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                source_path="/doc.pdf",
                snippet="text",
                score=1.5,
            )

    def test_citation_negative_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            Citation(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                source_path="/doc.pdf",
                snippet="text",
                score=-0.1,
            )

    def test_citation_valid_score(self) -> None:
        c = Citation(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            source_path="/doc.pdf",
            snippet="relevant excerpt",
            score=0.85,
        )
        assert c.score == pytest.approx(0.85)


class TestSession:
    def test_session_message_role_validation(self) -> None:
        msg = Message(
            session_id=uuid.uuid4(),
            role=MessageRole.USER,
            content="Hello!",
        )
        assert msg.role == MessageRole.USER

    def test_session_message_invalid_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            Message(
                session_id=uuid.uuid4(),
                role="invalid_role",  # type: ignore[arg-type]
                content="oops",
            )

    def test_session_constructs_with_defaults(self) -> None:
        s = Session()
        assert isinstance(s.id, uuid.UUID)
        assert s.user_id is None

    def test_session_frozen_mutation_raises(self) -> None:
        s = Session()
        with pytest.raises(ValidationError):
            s.user_id = "user-42"  # type: ignore[misc]
