"""FastAPI application — entry point for the RAG REST API.

Exposes:
    POST /chat         — SSE streaming chat.
    POST /chat/sync    — Synchronous JSON chat.
    POST /ingest       — Document ingestion.
    GET  /sessions/{id} — Session history.
    GET  /health       — Liveness / readiness probe.
    GET  /metrics      — Prometheus exposition format (mounted ASGI app).
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text

from src.infrastructure import di
from src.interface.api.deps import (
    AgentDep,
    IngestUseCaseDep,
    SessionRepoDep,
    SettingsDep,
)
from src.interface.api.middleware import RequestLoggingMiddleware
from src.interface.api.schemas import (
    ChatRequest,
    CitationOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MessageOut,
    SessionResponse,
    SyncChatResponse,
)
from src.shared.metrics import metrics_asgi_app

__all__ = ["app"]

log = structlog.get_logger(__name__)

try:
    _APP_VERSION = importlib.metadata.version("ai-assistant")
except importlib.metadata.PackageNotFoundError:
    _APP_VERSION = "0.1.0"

_IS_PRODUCTION = os.getenv("RAG_APP_ENV", "development").lower() == "production"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Warm stateful singletons on startup; dispose engine on shutdown."""
    log.info("api.startup: warming embedder and reranker")
    di.build_embedder()
    di.build_reranker()
    log.info("api.startup: ready")
    yield
    log.info("api.shutdown: disposing database engine")
    await di.build_engine().dispose()
    log.info("api.shutdown: done")


app = FastAPI(
    title="RAG Assistant API",
    version=_APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.mount("/metrics", metrics_asgi_app())


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    detail: dict[str, Any] = {"detail": str(exc)}
    if not _IS_PRODUCTION:
        detail["traceback"] = traceback.format_exc()
    log.error("unhandled_exception", exc_info=exc, path=request.url.path)
    return JSONResponse(status_code=500, content=detail)


@app.post("/chat")
async def chat_sse(body: ChatRequest, agent: AgentDep) -> StreamingResponse:
    """Stream chat response as Server-Sent Events."""
    state: dict[str, Any] = {
        "query": body.query,
        "session_id": body.session_id,
    }
    if body.language:
        state["language"] = body.language

    async def _event_generator() -> AsyncGenerator[str, None]:
        result: dict[str, Any] = await agent.ainvoke(state)
        answer = result.get("final_answer")
        if answer is not None:
            for word in answer.text.split():
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            citations = [
                CitationOut(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    page=c.page,
                    marker=c.marker,
                ).model_dump()
                for c in answer.citations
            ]
            yield f"data: {json.dumps({'done': True, 'citations': citations})}\n\n"
        else:
            yield f"data: {json.dumps({'done': True, 'citations': []})}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


@app.post("/chat/sync", response_model=SyncChatResponse)
async def chat_sync(body: ChatRequest, agent: AgentDep) -> SyncChatResponse:
    """Return the full answer as a single JSON response."""
    state: dict[str, Any] = {
        "query": body.query,
        "session_id": body.session_id,
    }
    if body.language:
        state["language"] = body.language

    result: dict[str, Any] = await agent.ainvoke(state)
    answer = result.get("final_answer")

    if answer is None:
        raise HTTPException(status_code=500, detail="Agent returned no answer")

    citations = [
        CitationOut(
            chunk_id=c.chunk_id,
            source=c.source,
            page=c.page,
            marker=c.marker,
        )
        for c in answer.citations
    ]
    session_id = body.session_id or ""
    return SyncChatResponse(
        text=answer.text,
        citations=citations,
        language=answer.language,
        tokens_in=answer.tokens_in,
        tokens_out=answer.tokens_out,
        session_id=session_id,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, ingest_uc: IngestUseCaseDep) -> IngestResponse:
    """Ingest documents from the given filesystem path."""
    report = await ingest_uc.execute(
        Path(body.path),
        language_hint=body.language_hint,
    )
    return IngestResponse(
        files=report.files_processed,
        chunks=report.chunks_created,
        document_ids=[],
    )


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, repo: SessionRepoDep) -> SessionResponse:
    """Return message history for a session by ULID."""
    messages = await repo.get_history(session_id)
    return SessionResponse(
        session_id=session_id,
        messages=[
            MessageOut(role=m.role.value, content=m.content) for m in messages
        ],
    )


@app.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Return liveness/readiness status for the API and its dependencies."""
    db_status = "ok"
    try:
        engine = di.build_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        db_status = "fail"

    langfuse_status = "disabled" if not settings.langfuse.enabled else "ok"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        version=_APP_VERSION,
        db=db_status,
        langfuse=langfuse_status,
    )
