from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config.settings import Settings
from src.infrastructure.di import (
    _reset_caches,
    build_agent,
    build_embedder,
    build_llm,
    build_reranker,
    build_settings,
)
from src.infrastructure.llm.gemini import GeminiLLM
from src.infrastructure.llm.openai import OpenAILLM


@pytest.fixture(autouse=True)
def _isolate_di_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM__API_KEY", "test")
    _reset_caches()
    yield  # type: ignore[misc]
    _reset_caches()


def test_build_settings_returns_settings_instance() -> None:
    result = build_settings()
    assert isinstance(result, Settings)


def test_build_settings_is_cached() -> None:
    s1 = build_settings()
    s2 = build_settings()
    assert s1 is s2


def test_build_embedder_is_singleton() -> None:
    e1 = build_embedder()
    e2 = build_embedder()
    assert e1 is e2


def test_build_reranker_is_singleton() -> None:
    r1 = build_reranker()
    r2 = build_reranker()
    assert r1 is r2


def test_build_llm_default_is_gemini() -> None:
    result = build_llm()
    assert isinstance(result, GeminiLLM)


def test_build_llm_openai_via_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM__PROVIDER", "openai")
    _reset_caches()
    result = build_llm(Settings())
    assert isinstance(result, OpenAILLM)


def test_build_llm_gemini_via_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM__PROVIDER", "gemini")
    _reset_caches()
    result = build_llm(Settings())
    assert isinstance(result, GeminiLLM)


def test_build_llm_unknown_provider_raises() -> None:
    mock_settings = MagicMock()
    mock_settings.llm.provider = "anthropic"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm(mock_settings)


def test_build_agent_returns_compiled_graph() -> None:
    graph = build_agent(Settings())
    assert hasattr(graph, "ainvoke")


def test_reset_caches_produces_fresh_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    s1 = build_settings()
    _reset_caches()
    monkeypatch.setenv("RAG_LLM__PROVIDER", "openai")
    s2 = build_settings()
    assert s1 is not s2
    assert s2.llm.provider == "openai"
