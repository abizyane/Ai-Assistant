"""Internationalisation helpers for the web UI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["SUPPORTED_LANGUAGES", "load_translations"]

_logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "fr": "Français",
    "ar": "العربية",
}

_CACHE: dict[str, dict[str, str]] = {}


def load_translations(lang: str) -> dict[str, Any]:
    """Load translation strings for *lang*, falling back to ``"en"``.

    Args:
        lang: BCP-47 language code (``"en"``, ``"fr"``, ``"ar"``).

    Returns:
        Mapping of translation key → localised string.
    """
    if lang in _CACHE:
        return _CACHE[lang]

    target = lang if lang in SUPPORTED_LANGUAGES else "en"
    path = Path(__file__).parent / f"{target}.json"

    if not path.exists():
        _logger.warning("Translation file not found for %r, falling back to 'en'", target)
        path = Path(__file__).parent / "en.json"

    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    _CACHE[target] = data
    if lang != target:
        _CACHE[lang] = data
    return data
