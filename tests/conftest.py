from __future__ import annotations

import importlib
from collections.abc import AsyncIterator

import pytest


# Keep these minimal and aligned with domain ports
class FakeLLM:
    async def generate(self, request):
        from src.domain.ports.dto import GenerationResult

        return GenerationResult(
            text="", input_tokens=0, output_tokens=0, model="fake", finish_reason="stop"
        )

    async def stream(self, request) -> AsyncIterator[str]:
        if False:
            yield
        return


class FakeEmbedder:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1024 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1024

    @property
    def dimension(self) -> int:
        return 1024


class FakeVectorStore:
    async def upsert(self, chunks):
        return len(chunks)

    async def search(self, query_vector, top_k, filters=None):
        return []

    async def hybrid_search(self, query_vector, query_text, top_k, filters=None):
        return []

    async def delete_by_document(self, document_id):
        return 0

    async def count(self):
        return 0


class FakeReranker:
    async def rerank(self, request):
        return request.chunks


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # ensure no real API keys are used
    monkeypatch.setenv("RAG_LLM__API_KEY", "test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _reset_contextvars_and_di() -> None:
    # reset tracing contextvars
    try:
        from src.shared.tracing import current_session_id, current_trace_id

        current_trace_id.set(None)
        current_session_id.set(None)
    except Exception:
        pass
    # reset DI caches if present
    try:
        di = importlib.import_module("src.infrastructure.di")
        if hasattr(di, "_reset_caches"):
            di._reset_caches()
    except Exception:
        pass


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def fake_reranker() -> FakeReranker:
    return FakeReranker()
