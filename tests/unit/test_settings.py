"""Unit tests for settings module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.settings import (
    AgentSettings,
    LLMSettings,
    get_settings,
)


def test_llm_settings_defaults() -> None:
    """LLMSettings uses sensible defaults."""
    s = LLMSettings(api_key="test_key")
    assert s.provider == "gemini"
    assert s.temperature == 0.1
    assert s.max_tokens == 4096


def test_agent_settings_defaults() -> None:
    """AgentSettings defaults match loop-bound decisions."""
    s = AgentSettings()
    assert s.max_rewrite_attempts == 1
    assert s.max_regen_attempts == 1
    assert s.max_steps == 15


def test_llm_api_key_is_secret() -> None:
    """API key is stored as SecretStr (not exposed in repr)."""
    s = LLMSettings(api_key="my_secret_key")
    assert "my_secret_key" not in repr(s)
    assert s.api_key.get_secret_value() == "my_secret_key"


def test_settings_llm_temperature_validation() -> None:
    """Temperature outside [0, 2] raises ValidationError."""
    with pytest.raises(ValidationError):
        LLMSettings(api_key="key", temperature=3.5)


def test_get_settings_lru_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() returns cached instance on repeated calls."""
    monkeypatch.setenv("RAG_LLM__API_KEY", "test_key_for_cache")
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
