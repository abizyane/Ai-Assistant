"""Typer CLI entry point for rag-cli.

Commands:
    ingest <path>       — Ingest documents into the vector store.
    chat [query]        — Chat with the agent (interactive REPL if no query given).
    evaluate <dataset>  — Run Ragas evaluation over a JSONL golden-set.
    sessions            — Manage chat sessions (subcommands: list, show).
    health              — Check database connectivity.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.prompt import Prompt
from rich.table import Table
from sqlalchemy import text

from src.application.use_cases.evaluate import check_thresholds
from src.infrastructure.di import (
    build_agent,
    build_engine,
    build_evaluate_use_case,
    build_ingest_use_case,
    build_session_repo,
    build_settings,
)
from src.interface.cli._render import (
    console,
    make_eval_table,
    make_ingest_summary_table,
    make_progress,
    make_sessions_table,
    render_answer,
    render_error,
    render_success,
)

__all__ = ["app"]

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="[bold]rag-cli[/bold] — agentic RAG assistant for 1337 Coding School.",
)

sessions_app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Manage chat sessions.",
)
app.add_typer(sessions_app, name="sessions")

_EXIT_WORDS: frozenset[str] = frozenset({"bye", "exit", "quit"})


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Directory or file to ingest."),  # noqa: B008
    language_hint: str | None = typer.Option(
        None,
        "--language-hint",
        "-l",
        help="Override language detection (e.g. 'en', 'fr').",
    ),
) -> None:
    """Ingest documents from [cyan]PATH[/cyan] into the vector store.

    Supports PDF, Markdown, and HTML files (discovered recursively for directories).
    Shows a per-file progress bar and a summary table on completion.
    """
    settings = build_settings()
    use_case = build_ingest_use_case(settings)

    _supported = (".pdf", ".md", ".html")
    if path.is_dir():
        source_files: list[Path] = sorted(
            f for suffix in _supported for f in path.rglob(f"*{suffix}")
        )
    elif path.is_file():
        source_files = [path]
    else:
        render_error(f"Path does not exist: {path}")
        raise typer.Exit(code=1)

    if not source_files:
        console.print("[yellow]No supported files found (.pdf, .md, .html).[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"[bold]Ingesting {len(source_files)} file(s) from[/bold] {path}")

    with make_progress() as progress:
        task_id = progress.add_task("Ingesting\u2026", total=len(source_files))

        async def _ingest_run() -> object:
            report = await use_case.execute(path, language_hint=language_hint)
            progress.update(task_id, completed=len(source_files))
            return report

        report = asyncio.run(_ingest_run())

    console.print(
        make_ingest_summary_table(
            files_processed=report.files_processed,  # type: ignore[attr-defined]
            files_skipped=report.files_skipped,  # type: ignore[attr-defined]
            chunks_created=report.chunks_created,  # type: ignore[attr-defined]
            duration_seconds=report.duration_seconds,  # type: ignore[attr-defined]
            errors=report.errors,  # type: ignore[attr-defined]
        )
    )

    if report.errors:  # type: ignore[attr-defined]
        for file_path, err_msg in report.errors:  # type: ignore[attr-defined]
            render_error(f"{file_path}: {err_msg}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command()
def chat(
    query: str | None = typer.Argument(
        None,
        help="Single query to answer. Omit to start an interactive REPL.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Resume an existing session by ULID.",
    ),
) -> None:
    """Chat with the RAG agent.

    If [cyan]QUERY[/cyan] is omitted, an interactive REPL starts.
    Type [bold]exit[/bold], [bold]quit[/bold], or [bold]bye[/bold] to leave.
    """
    settings = build_settings()
    agent = build_agent(settings)

    async def _ask(question: str) -> None:
        state: dict = await agent.ainvoke(  # type: ignore[type-arg]
            {"query": question, "session_id": session_id}
        )
        answer = state.get("final_answer")
        if answer is None:
            render_error("No answer generated.")
            return
        render_answer(answer.text, answer.citations)

    if query is not None:
        asyncio.run(_ask(query))
        return

    console.print("[bold green]rag-cli chat[/bold green] \u2014 type [bold]exit[/bold] to quit\n")
    while True:
        try:
            user_input = Prompt.ask("[bold]You[/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not user_input:
            continue
        if user_input.lower() in _EXIT_WORDS:
            console.print("[dim]Goodbye.[/dim]")
            break
        asyncio.run(_ask(user_input))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    dataset: Path = typer.Argument(..., help="Path to JSONL evaluation dataset."),  # noqa: B008
    sample: int | None = typer.Option(
        None,
        "--sample",
        "-n",
        help="Evaluate only the first N rows.",
    ),
) -> None:
    """Run Ragas evaluation over [cyan]DATASET[/cyan] and report metric scores.

    Exits with code [bold]1[/bold] if any metric falls below its configured threshold.
    """
    settings = build_settings()
    use_case = build_evaluate_use_case(settings)

    console.print(f"[bold]Evaluating dataset:[/bold] {dataset}")

    async def _eval_run() -> object:
        return await use_case.execute(dataset, sample=sample)

    report = asyncio.run(_eval_run())

    thresholds = {
        "faithfulness": settings.eval.faithfulness,
        "answer_relevancy": settings.eval.answer_relevancy,
        "context_precision": settings.eval.context_precision,
        "context_recall": settings.eval.context_recall,
        "answer_correctness": settings.eval.answer_correctness,
    }
    console.print(make_eval_table(report.aggregate, thresholds))  # type: ignore[attr-defined]

    failures = check_thresholds(report, settings)  # type: ignore[arg-type]
    if failures:
        for metric, score, threshold in failures:  # type: ignore[misc]
            render_error(f"{metric}: {score:.4f} < threshold {threshold:.4f}")
        raise typer.Exit(code=1)

    render_success(f"All thresholds passed ({report.sample_size} rows evaluated).")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@app.command()
def health() -> None:
    """Check database connectivity and overall system health.

    Exits with code [bold]1[/bold] on failure.
    """
    engine = build_engine()

    async def _health_check() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    try:
        asyncio.run(_health_check())
        render_success("Database reachable \u2014 system healthy.")
    except Exception as exc:
        render_error(f"Health check failed: {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# sessions subcommands
# ---------------------------------------------------------------------------


@sessions_app.command("list")
def sessions_list(
    user_id: str | None = typer.Option(
        None,
        "--user-id",
        "-u",
        help="Filter sessions by user ID.",
    ),
) -> None:
    """List all chat sessions, ordered by last activity.

    Use [bold]--user-id[/bold] to filter by a specific user.
    """
    repo = build_session_repo()

    async def _list_run() -> list:  # type: ignore[type-arg]
        return await repo.list_sessions(user_id=user_id)

    sessions = asyncio.run(_list_run())

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    console.print(make_sessions_table(sessions))


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="ULID string identifying the session."),
) -> None:
    """Show the message history for a session by [cyan]SESSION_ID[/cyan]."""
    repo = build_session_repo()

    async def _show_run() -> list:  # type: ignore[type-arg]
        return await repo.get_history(session_id)

    messages = asyncio.run(_show_run())

    if not messages:
        console.print(f"[yellow]No messages found for session {session_id}.[/yellow]")
        return

    table = Table(
        title=f"Session {session_id}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Role", style="bold", no_wrap=True)
    table.add_column("Content")
    table.add_column("Created", no_wrap=True)

    for msg in messages:
        created = str(getattr(msg, "created_at", ""))[:19]
        role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        table.add_row(role_val, msg.content, created)

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Typer CLI entry point invoked by the ``rag-cli`` script."""
    app()


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[func-returns-value]
