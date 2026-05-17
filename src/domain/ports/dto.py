"""Data Transfer Objects for cross-port communication in the domain layer."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRequest(BaseModel):
    """Request to embed a list of texts into vector representations."""

    model_config = ConfigDict(frozen=True)

    texts: list[str]
    model: str


class EmbeddingResult(BaseModel):
    """Result of a batch text embedding operation."""

    model_config = ConfigDict(frozen=True)

    embeddings: list[list[float]]
    model: str
    dimension: int


class RetrievalQuery(BaseModel):
    """Hybrid retrieval query combining dense and sparse search parameters."""

    model_config = ConfigDict(frozen=True)

    text: str
    language: str
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_rerank: int = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store with an associated relevance score."""

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankRequest(BaseModel):
    """Request to rerank a set of retrieved chunks against a query."""

    model_config = ConfigDict(frozen=True)

    query: str
    chunks: list[RetrievedChunk]
    top_k: int = 5


class GenerationRequest(BaseModel):
    """Request to generate a text response via an LLM."""

    model_config = ConfigDict(frozen=True)

    messages: list[dict[str, str]]
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024


class GenerationResult(BaseModel):
    """Result of an LLM text generation call."""

    model_config = ConfigDict(frozen=True)

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str = "stop"


class TraceContext(BaseModel):
    """Context object representing an active observability trace span."""

    model_config = ConfigDict(frozen=True)

    span_id: str
    trace_id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawDocument(BaseModel):
    """Raw document content loaded from a source path before chunking."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    content: str
    language: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkContent(BaseModel):
    """A single text chunk produced by a chunker, prior to embedding."""

    model_config = ConfigDict(frozen=True)

    content: str
    position: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkWithEmbedding(BaseModel):
    """A text chunk paired with its dense embedding, ready for vector store insertion."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    document_id: UUID
    content: str
    embedding: list[float]
    position: int
    token_count: int = 0
    source_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRequest(BaseModel):
    """Request to evaluate RAG pipeline outputs against ground-truth data."""

    model_config = ConfigDict(frozen=True)

    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Aggregated evaluation metrics from an evaluation run."""

    model_config = ConfigDict(frozen=True)

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    per_item: list[dict[str, Any]] = Field(default_factory=list)


class IngestionReport(BaseModel):
    """Summary of a completed document ingestion pipeline run."""

    model_config = ConfigDict(frozen=True)

    files_processed: int
    files_skipped: int
    chunks_created: int
    duration_seconds: float
    errors: list[tuple[str, str]] = Field(default_factory=list)
