from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.interface.web.i18n import SUPPORTED_LANGUAGES, load_translations

_EXPECTED_KEYS = {
    "app_title",
    "app_subtitle",
    "nav_chat",
    "nav_ingest",
    "nav_sessions",
    "nav_eval",
    "sidebar_language",
    "sidebar_session_id_label",
    "sidebar_session_id_empty",
    "sidebar_clear",
    "sidebar_new_session",
    "chat_input_placeholder",
    "chat_thinking",
    "chat_no_messages",
    "citations_expander",
    "citation_source",
    "citation_page",
    "citation_score",
    "langfuse_trace",
    "ingest_title",
    "ingest_upload_label",
    "ingest_button",
    "ingest_success",
    "ingest_error",
    "sessions_title",
    "sessions_select",
    "sessions_no_sessions",
    "sessions_role_user",
    "sessions_role_assistant",
    "eval_title",
    "eval_no_runs",
    "eval_metric_score",
    "eval_threshold",
    "eval_docs_link",
    "error_api_unavailable",
    "error_generic",
    "lang_en",
    "lang_fr",
    "lang_ar",
    "citations_header",
    "error_network",
}


@pytest.mark.parametrize("lang", list(SUPPORTED_LANGUAGES.keys()))
def test_translation_keys_parity(lang: str) -> None:
    translations = load_translations(lang)
    assert set(translations.keys()) == _EXPECTED_KEYS, (
        f"Language '{lang}' has mismatched keys: "
        f"missing={_EXPECTED_KEYS - set(translations.keys())}, "
        f"extra={set(translations.keys()) - _EXPECTED_KEYS}"
    )


@pytest.mark.parametrize("lang", list(SUPPORTED_LANGUAGES.keys()))
def test_no_empty_values(lang: str) -> None:
    translations = load_translations(lang)
    empty = [k for k, v in translations.items() if not v]
    assert not empty, f"Language '{lang}' has empty values for keys: {empty}"


def test_unknown_lang_falls_back_to_en() -> None:
    translations = load_translations("xx")
    en_translations = load_translations("en")
    assert translations == en_translations


def test_supported_languages_has_three_entries() -> None:
    assert set(SUPPORTED_LANGUAGES.keys()) == {"en", "fr", "ar"}


@pytest.mark.parametrize("lang", list(SUPPORTED_LANGUAGES.keys()))
def test_json_files_are_valid(lang: str) -> None:
    i18n_dir = Path(__file__).parent.parent.parent.parent / "src" / "interface" / "web" / "i18n"
    path = i18n_dir / f"{lang}.json"
    assert path.exists(), f"Missing translation file: {path}"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)
