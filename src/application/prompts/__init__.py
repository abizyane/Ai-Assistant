"""Jinja2 prompt template loader for the application layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

__all__ = ["PromptTemplateLoader"]


class PromptTemplateLoader:
    """Loads and renders Jinja2 ``.j2`` templates from a directory.

    By default loads templates from this package directory. ``StrictUndefined``
    is enabled so missing template variables raise immediately rather than
    rendering empty strings (helps catch prompt-construction bugs early).
    """

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        """Initialize the loader.

        Args:
            templates_dir: Directory holding ``.j2`` templates. Defaults to the
                directory of this module.
        """
        base = Path(templates_dir) if templates_dir else Path(__file__).parent
        self._env = Environment(
            loader=FileSystemLoader(str(base)),
            autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **context: Any) -> str:  # noqa: ANN401
        """Render a template by name with the given context variables.

        Args:
            template_name: Filename of the template (e.g. ``answer_with_citations.j2``).
            **context: Variables passed to the template.

        Returns:
            The rendered prompt string.
        """
        template = self._env.get_template(template_name)
        return template.render(**context)
