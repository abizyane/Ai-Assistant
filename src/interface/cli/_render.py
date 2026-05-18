"""Rich rendering helpers for the rag-cli Typer application.

All functions accept domain objects and return or print Rich renderables.
No ``print()`` calls — always use the module-level ``console``.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

__all__ = [
    "console",
    "make_citation_table",
    "make_eval_table",
    "make_ingest_summary_table",
    "make_progress",
    "make_sessions_table",
    "render_answer",
    "render_error",
    "render_success",
]

console = Console()


def make_progress() -> Progress:
    """Return a Rich Progress instance with spinner, bar, and elapsed time.

    Returns:
        Configured ``Progress`` context manager.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def render_answer(text: str, citations: list[Any]) -> None:
    """Print the answer text and a citations table to the console.

    Args:
        text: LLM-generated answer text.
        citations: List of :class:`~src.domain.entities.citation.Citation` objects.
    """
    console.print()
    syntax = Syntax(text, "markdown", theme="monokai", word_wrap=True)
    console.print(Panel(syntax, title="[bold green]Answer[/bold green]", border_style="green"))
    if citations:
        console.print(make_citation_table(citations))
    console.print()


def make_citation_table(citations: list[Any]) -> Table:
    """Build a Rich Table of citations.

    Args:
        citations: List of :class:`~src.domain.entities.citation.Citation` objects.

    Returns:
        ``Table`` with columns: marker | source | page.
    """
    table = Table(title="Citations", show_header=True, header_style="bold cyan")
    table.add_column("Marker", style="yellow", no_wrap=True)
    table.add_column("Source", style="dim")
    table.add_column("Page", justify="right")
    for c in citations:
        page_str = str(c.page) if c.page is not None else "—"
        table.add_row(c.marker, c.source, page_str)
    return table


def make_ingest_summary_table(
    files_processed: int,
    files_skipped: int,
    chunks_created: int,
    duration_seconds: float,
    errors: list[tuple[str, str]],
) -> Table:
    """Build a Rich summary Table for the ingest command.

    Args:
        files_processed: Number of files successfully ingested.
        files_skipped: Number of files skipped (already ingested).
        chunks_created: Total text chunks created.
        duration_seconds: Wall-clock duration of the ingestion run.
        errors: List of ``(path, error_message)`` tuples for failed files.

    Returns:
        Rich ``Table`` with ingestion statistics.
    """
    table = Table(title="Ingestion Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Files processed", str(files_processed))
    table.add_row("Files skipped", str(files_skipped))
    table.add_row("Chunks created", str(chunks_created))
    table.add_row("Duration (s)", f"{duration_seconds:.2f}")
    table.add_row("Errors", str(len(errors)))
    return table


def make_eval_table(aggregate: dict[str, float], thresholds: dict[str, float]) -> Table:
    """Build a Rich Table of evaluation metric scores vs thresholds.

    Args:
        aggregate: Dict of metric_name → mean score.
        thresholds: Dict of metric_name → minimum required score.

    Returns:
        Rich ``Table`` with columns: metric | score | threshold | status.
    """
    table = Table(title="Evaluation Results", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")
    for metric, score in sorted(aggregate.items()):
        threshold = thresholds.get(metric, 0.0)
        passed = score >= threshold
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(metric, f"{score:.4f}", f"{threshold:.4f}", status)
    return table


def make_sessions_table(sessions: list[dict[str, Any]]) -> Table:
    """Build a Rich Table listing session summaries.

    Args:
        sessions: List of session summary dicts with keys
            ``id``, ``user_id``, ``created_at``, ``last_active``, ``message_count``.

    Returns:
        Rich ``Table`` with one row per session.
    """
    table = Table(title="Chat Sessions", show_header=True, header_style="bold yellow")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("User")
    table.add_column("Created")
    table.add_column("Last Active")
    table.add_column("Messages", justify="right")
    for s in sessions:
        created = str(s.get("created_at", ""))[:19]
        last_active = str(s.get("last_active", ""))[:19]
        table.add_row(
            str(s.get("id", "")),
            str(s.get("user_id") or "—"),
            created,
            last_active,
            str(s.get("message_count", 0)),
        )
    return table


def render_success(message: str) -> None:
    """Print a green success panel to the console.

    Args:
        message: Success message text.
    """
    console.print(Panel(f"[green]{message}[/green]", border_style="green"))


def render_error(message: str) -> None:
    """Print a red error panel to the console.

    Args:
        message: Error message text.
    """
    console.print(Panel(f"[red]{message}[/red]", title="Error", border_style="red"))
