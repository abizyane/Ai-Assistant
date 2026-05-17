"""FastAPI ``Depends``-compatible dependency factories.

Each getter is a thin wrapper that delegates to the DI composition root
(``src.infrastructure.di``), making every dependency easily monkeypatchable
in tests without touching the DI module itself.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from src.application.use_cases.ingest_documents import IngestDocumentsUseCase
from src.config.settings import Settings
from src.infrastructure import di
from src.infrastructure.persistence.session_repo import SessionRepository

__all__ = [
    "AgentDep",
    "IngestUseCaseDep",
    "SessionRepoDep",
    "SettingsDep",
    "get_agent",
    "get_ingest_use_case",
    "get_session_repo",
    "get_settings",
]


def get_settings() -> Settings:
    """Return the singleton application settings."""
    return di.build_settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_agent(settings: SettingsDep) -> CompiledStateGraph:
    """Build and return the compiled agentic RAG graph."""
    return di.build_agent(settings)


AgentDep = Annotated[CompiledStateGraph, Depends(get_agent)]


def get_ingest_use_case(settings: SettingsDep) -> IngestDocumentsUseCase:
    """Build and return the document ingestion use case."""
    return di.build_ingest_use_case(settings)


IngestUseCaseDep = Annotated[IngestDocumentsUseCase, Depends(get_ingest_use_case)]


def get_session_repo(settings: SettingsDep) -> SessionRepository:
    """Build and return the session repository adapter."""
    return di.build_session_repo(settings)


SessionRepoDep = Annotated[SessionRepository, Depends(get_session_repo)]
