"""DI composition root — plain build_* factory functions, no framework.

All application wiring lives here and *nowhere else* (composition-root
principle).  Each ``build_*`` function takes an optional ``Settings`` argument
so callers can supply a custom settings object; when omitted, the module-level
``build_settings()`` singleton is used.

Stateful singletons (engine, embedder, reranker) are cached via ``@lru_cache``
on private no-arg helpers that call ``build_settings()`` internally, making them
hashable and safe to use with ``lru_cache``.

Call :func:`_reset_caches` in tests before changing environment variables so
the next ``build_*`` call picks up the new values.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine

from src.application.agent.graph import build_agent_graph
from src.application.prompts import PromptTemplateLoader
from src.application.use_cases.evaluate import EvaluateUseCase, default_agent_runner
from src.application.use_cases.generate_answer import GenerateAnswerUseCase
from src.application.use_cases.ingest_documents import IngestDocumentsUseCase
from src.application.use_cases.retrieve import RetrieveUseCase
from src.config.settings import Settings
from src.domain.ports.llm import LLMPort
from src.infrastructure.chunking.semantic_chunker import SemanticChunker
from src.infrastructure.embeddings.multilingual_e5_embedder import MultilingualE5Embedder
from src.infrastructure.llm.gemini import GeminiLLM
from src.infrastructure.llm.openai import OpenAILLM
from src.infrastructure.loading.docling_loader import DoclingLoader
from src.infrastructure.observability.langfuse_tracer import LangfuseTracer
from src.infrastructure.persistence.engine import create_engine, create_session_factory
from src.infrastructure.persistence.session_repo import SessionRepository
from src.infrastructure.persistence.vector_store import PGVectorStore
from src.infrastructure.reranking.bge_reranker import BGEReranker

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

__all__ = [
    "_reset_caches",
    "build_agent",
    "build_chunker",
    "build_embedder",
    "build_engine",
    "build_evaluate_use_case",
    "build_generate_use_case",
    "build_ingest_use_case",
    "build_llm",
    "build_loader",
    "build_logger",
    "build_reranker",
    "build_retrieve_use_case",
    "build_session_repo",
    "build_settings",
    "build_tracer",
    "build_vector_store",
]

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Cached settings singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def build_settings() -> Settings:
    """Return singleton Settings loaded from environment / .env.

    Returns:
        Cached ``Settings`` instance.
    """
    return Settings()


# ---------------------------------------------------------------------------
# Stateful singleton adapters (no-arg internal + public wrapper)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cached_engine() -> AsyncEngine:
    """Build and cache the async SQLAlchemy engine (internal)."""
    settings = build_settings()
    return create_engine(settings)


def build_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the singleton async database engine.

    The engine is a process-level singleton cached via ``@lru_cache``.  The
    ``settings`` argument is accepted for API consistency but **not** used —
    the engine is always constructed from ``build_settings()``.

    Args:
        settings: Accepted but unused; engine uses ``build_settings()``
            internally.

    Returns:
        Singleton ``AsyncEngine`` instance.
    """
    del settings
    return _cached_engine()


@lru_cache(maxsize=1)
def _cached_embedder() -> MultilingualE5Embedder:
    """Build and cache the multilingual-e5-small embedder (internal)."""
    settings = build_settings()
    return MultilingualE5Embedder(settings)


def build_embedder(settings: Settings | None = None) -> MultilingualE5Embedder:
    """Return the singleton multilingual-e5-small dense embedder.

    The model is loaded lazily on the first :meth:`embed_texts` call and then
    reused for the lifetime of the process.

    Args:
        settings: Accepted but unused; embedder uses ``build_settings()``
            internally.

    Returns:
        Singleton ``MultilingualE5Embedder`` instance.
    """
    del settings
    return _cached_embedder()


@lru_cache(maxsize=1)
def _cached_reranker() -> BGEReranker:
    """Build and cache the BGE cross-encoder reranker (internal)."""
    settings = build_settings()
    return BGEReranker(settings)


def build_reranker(settings: Settings | None = None) -> BGEReranker:
    """Return the singleton BGE cross-encoder reranker.

    The model is loaded lazily on the first :meth:`rerank` call and then
    reused for the lifetime of the process.

    Args:
        settings: Accepted but unused; reranker uses ``build_settings()``
            internally.

    Returns:
        Singleton ``BGEReranker`` instance.
    """
    del settings
    return _cached_reranker()


# ---------------------------------------------------------------------------
# Light (uncached) adapters
# ---------------------------------------------------------------------------


def build_llm(settings: Settings | None = None) -> LLMPort:
    """Return a fresh LLM adapter for the configured provider.

    Reads ``settings.llm.provider`` and instantiates either ``GeminiLLM``
    (default) or ``OpenAILLM``.  The result is **not** cached because LLM
    adapters are cheap to construct and provider changes should take immediate
    effect after ``_reset_caches()``.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        A ``LLMPort`` implementation for the configured provider.

    Raises:
        ValueError: If the provider name is unrecognised.
    """
    s = settings or build_settings()
    provider = s.llm.provider.lower()
    if provider in {"openai", "openrouter", "nvidia"}:
        if provider == "openrouter" and not s.llm.base_url:
            s.llm.base_url = "https://openrouter.ai/api/v1"
        elif provider == "nvidia" and not s.llm.base_url:
            s.llm.base_url = "https://integrate.api.nvidia.com/v1"
        return OpenAILLM(s)
    if provider == "gemini":
        return GeminiLLM(s)
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Expected 'gemini', 'openai', 'openrouter', or 'nvidia'."
    )


def build_tracer(settings: Settings | None = None) -> LangfuseTracer:
    """Return a Langfuse tracer adapter.

    If Langfuse credentials are absent or the SDK is not installed the tracer
    degrades gracefully to a no-op (see ``LangfuseTracer`` implementation).

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``LangfuseTracer`` instance (may be a no-op if unconfigured).
    """
    s = settings or build_settings()
    return LangfuseTracer(
        host=s.langfuse.host,
        public_key=s.langfuse.public_key,
        secret_key=s.langfuse.secret_key.get_secret_value(),
    )


def build_logger() -> logging.Logger:
    """Return the application-level standard-library logger.

    Returns:
        ``logging.Logger`` named ``"rag"``.
    """
    return logging.getLogger("rag")


def build_loader(settings: Settings | None = None) -> DoclingLoader:
    """Return a Docling document loader adapter.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``DoclingLoader`` instance.
    """
    s = settings or build_settings()
    return DoclingLoader(s)


def build_chunker(settings: Settings | None = None) -> SemanticChunker:
    """Return a semantic chunker adapter.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``SemanticChunker`` instance.
    """
    s = settings or build_settings()
    return SemanticChunker(s)


def build_vector_store(settings: Settings | None = None) -> PGVectorStore:
    """Return a PGVector store adapter bound to the singleton engine.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``PGVectorStore`` instance backed by the singleton engine.
    """
    s = settings or build_settings()
    engine = _cached_engine()
    session_factory = create_session_factory(engine)
    return PGVectorStore(session_factory, s)


def build_session_repo(settings: Settings | None = None) -> SessionRepository:
    """Return a session repository adapter bound to the singleton engine.

    Args:
        settings: Application settings. Accepted for API consistency but unused
            because the engine is resolved via ``_cached_engine()``.

    Returns:
        ``SessionRepository`` instance.
    """
    del settings
    engine = _cached_engine()
    session_factory = create_session_factory(engine)
    return SessionRepository(session_factory)


# ---------------------------------------------------------------------------
# Use cases (uncached — cheap and stateless)
# ---------------------------------------------------------------------------


def build_ingest_use_case(settings: Settings | None = None) -> IngestDocumentsUseCase:
    """Build the document ingestion use case with all dependencies wired.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``IngestDocumentsUseCase`` ready for execution.
    """
    s = settings or build_settings()
    engine = _cached_engine()
    session_factory = create_session_factory(engine)
    return IngestDocumentsUseCase(
        loader=build_loader(s),
        chunker=build_chunker(s),
        embedder=build_embedder(s),
        vector_store=build_vector_store(s),
        session_repo=session_factory,
        tracer=build_tracer(s),
        logger=build_logger(),
    )


def build_retrieve_use_case(settings: Settings | None = None) -> RetrieveUseCase:
    """Build the retrieval use case with all dependencies wired.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``RetrieveUseCase`` ready for execution.
    """
    s = settings or build_settings()
    return RetrieveUseCase(
        embedder=build_embedder(s),
        vector_store=build_vector_store(s),
        reranker=build_reranker(s),
        settings=s,
    )


def build_generate_use_case(settings: Settings | None = None) -> GenerateAnswerUseCase:
    """Build the answer-generation use case with all dependencies wired.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``GenerateAnswerUseCase`` ready for execution.
    """
    s = settings or build_settings()
    return GenerateAnswerUseCase(
        llm=build_llm(s),
        prompt_template_loader=PromptTemplateLoader(),
        tracer=build_tracer(s),
        logger=build_logger(),
    )


def build_evaluate_use_case(settings: Settings | None = None) -> EvaluateUseCase:
    """Build the Ragas evaluation use case with all dependencies wired.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        ``EvaluateUseCase`` ready for execution.
    """
    s = settings or build_settings()
    graph = build_agent(s)
    runner = default_agent_runner(graph)
    return EvaluateUseCase(agent_runner=runner, settings=s)


def build_agent(settings: Settings | None = None) -> CompiledStateGraph:
    """Build and compile the full agentic RAG LangGraph.

    Wires retrieve \u2192 generate \u2192 verify_grounding into a compiled
    ``StateGraph``.  The returned graph is ready for ``ainvoke``/``astream``.

    Args:
        settings: Application settings. Defaults to ``build_settings()``.

    Returns:
        Compiled ``CompiledStateGraph`` instance.
    """
    s = settings or build_settings()
    retrieve_uc = build_retrieve_use_case(s)
    generate_uc = build_generate_use_case(s)
    grade_llm = build_llm(s)
    return build_agent_graph(
        retrieve_uc=retrieve_uc,
        generate_uc=generate_uc,
        grade_llm=grade_llm,
        settings=s,
    )


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _reset_caches() -> None:
    """Clear all lru_cache-d singletons.

    Call this in tests **before** each test that changes environment variables
    (e.g. ``RAG_LLM__PROVIDER``) so the next ``build_*`` call picks up the
    new values.  Should be paired with ``monkeypatch.setenv`` / ``monkeypatch``
    fixture cleanup so the environment is restored after each test.
    """
    build_settings.cache_clear()
    _cached_engine.cache_clear()
    _cached_embedder.cache_clear()
    _cached_reranker.cache_clear()
