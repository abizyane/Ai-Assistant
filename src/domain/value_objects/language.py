"""Language value object — validated ISO-639-1 language code."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

_VALID_CODES: frozenset[str] = frozenset(
    [
        "en",
        "fr",
        "ar",
        "es",
        "de",
        "it",
        "pt",
        "zh",
        "ja",
        "ko",
        "nl",
        "pl",
        "ru",
        "sv",
        "tr",
        "he",
        "fa",
        "ur",
        "hi",
        "vi",
        "id",
        "ms",
        "th",
        "uk",
        "cs",
        "ro",
        "hu",
        "fi",
        "da",
        "no",
    ]
)


class Language(BaseModel):
    """An immutable, validated ISO-639-1 language code."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=2, max_length=2, description="ISO-639-1 two-letter code")

    @field_validator("code")
    @classmethod
    def _validate_code(cls, v: str) -> str:
        """Validate code is a known ISO-639-1 language."""
        normalized = v.lower()
        if normalized not in _VALID_CODES:
            msg = f"Unknown ISO-639-1 language code: {v!r}"
            raise ValueError(msg)
        return normalized
