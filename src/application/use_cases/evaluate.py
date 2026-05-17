"""Evaluate use case — Ragas-based RAG quality measurement over a JSONL golden set."""

from __future__ import annotations

import json
import warnings
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

from datasets import Dataset

from src.config.settings import Settings
from src.domain.entities.answer import AnswerWithCitations
from src.domain.entities.evaluation import EvaluationReport
from src.shared.metrics import set_gauge
from src.shared.tracing import traced

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["EvaluateUseCase", "check_thresholds", "default_agent_runner"]

_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

_log = structlog.get_logger(__name__)


def default_agent_runner(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
) -> Callable[[str, str | None], Awaitable[AnswerWithCitations]]:
    """Build an ``agent_runner`` callable that wraps a compiled LangGraph.

    Args:
        graph: Compiled LangGraph ``StateGraph`` (returned by ``build_agent_graph``).

    Returns:
        An async callable ``(query, language) -> AnswerWithCitations``.
    """

    async def _run(query: str, language: str | None) -> AnswerWithCitations:
        state: dict[str, Any] = await graph.ainvoke({"query": query, "language": language or "en"})
        answer: AnswerWithCitations | None = state.get("final_answer")
        if answer is None:
            return AnswerWithCitations(
                text="No answer generated.",
                citations=[],
                language=language or "en",
                tokens_in=0,
                tokens_out=0,
            )
        return answer

    return _run


class EvaluateUseCase:
    """Run Ragas evaluation over a JSONL golden-set dataset.

    Each JSONL row must have::

        {"query": str, "ground_truth_answer": str,
         "ground_truth_contexts": list[str], "language": str | None}

    Results are written atomically to ``evals/runs/<timestamp>.json`` and
    Prometheus gauges ``rag_eval_<metric>`` are updated.
    """

    def __init__(
        self,
        agent_runner: Callable[[str, str | None], Awaitable[AnswerWithCitations]],
        settings: Settings,
    ) -> None:
        """Wire collaborators.

        Args:
            agent_runner: Async callable ``(query, language) -> AnswerWithCitations``.
                Use :func:`default_agent_runner` in production; inject an
                ``AsyncMock`` in tests.
            settings: Application settings (thresholds live under ``settings.eval``).
        """
        self._runner = agent_runner
        self._settings = settings

    @traced("use_case.evaluate")
    async def execute(
        self,
        dataset_path: Path,
        sample: int | None = None,
        output_dir: Path | None = None,
    ) -> EvaluationReport:
        """Evaluate the agent over *dataset_path* and return an :class:`EvaluationReport`.

        Args:
            dataset_path: Path to the JSONL file with evaluation rows.
            sample: If given, evaluate only the first *sample* rows.
            output_dir: Directory for JSON output. Defaults to ``evals/runs``.

        Returns:
            Frozen :class:`EvaluationReport` with per-question and aggregate scores.

        Raises:
            ValueError: If the dataset file is empty.
        """
        rows = _load_jsonl(dataset_path)
        if not rows:
            raise ValueError(f"Empty dataset: {dataset_path}")
        if sample is not None:
            rows = rows[:sample]

        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []

        for row in rows:
            query: str = row["query"]
            language: str | None = row.get("language")
            ground_truth_contexts: list[str] = row.get("ground_truth_contexts", [])

            answer = await self._runner(query, language)

            questions.append(query)
            answers.append(answer.text)
            contexts.append(ground_truth_contexts)
            ground_truths.append(row["ground_truth_answer"])

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        _log.info("evaluate.running_ragas", sample_size=len(rows))
        result = evaluate(dataset, metrics=_METRICS)

        per_row_scores: dict[str, list[float]] = {}
        for name in _METRIC_NAMES:
            try:
                raw = result[name]
                per_row_scores[name] = [float(s) if s is not None else 0.0 for s in raw]
            except (KeyError, TypeError):
                per_row_scores[name] = [0.0] * len(rows)

        per_question: list[dict[str, Any]] = []
        for i, _row in enumerate(rows):
            pq: dict[str, Any] = {
                "query": questions[i],
                "ground_truth": ground_truths[i],
                "prediction": answers[i],
                "contexts": contexts[i],
            }
            for name, scores in per_row_scores.items():
                pq[name] = scores[i]
            per_question.append(pq)

        aggregate: dict[str, float] = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in per_row_scores.items()
        }

        for name, value in aggregate.items():
            set_gauge(f"rag_eval_{name}", value)

        ts = datetime.now(UTC)
        report = EvaluationReport(
            timestamp=ts,
            dataset_path=str(dataset_path),
            sample_size=len(rows),
            per_question=per_question,
            aggregate=aggregate,
        )

        runs_dir = output_dir if output_dir is not None else Path("evals/runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
        out_path = runs_dir / f"{ts_str}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(report.to_json(), encoding="utf-8")
        tmp_path.rename(out_path)

        _log.info("evaluate.complete", report_path=str(out_path), aggregate=aggregate)
        return report


def check_thresholds(
    report: EvaluationReport,
    settings: Settings,
) -> tuple[bool, list[str]]:
    """Check whether all aggregate metrics clear configured thresholds.

    Args:
        report: Completed evaluation report.
        settings: Application settings containing ``eval`` thresholds.

    Returns:
        ``(passed, failed_metrics)`` — ``passed`` is ``True`` only when every
        metric in ``report.aggregate`` meets its threshold.
    """
    thresholds: dict[str, float] = {
        "faithfulness": settings.eval.faithfulness,
        "answer_relevancy": settings.eval.answer_relevancy,
        "context_precision": settings.eval.context_precision,
        "context_recall": settings.eval.context_recall,
        "answer_correctness": settings.eval.answer_correctness,
    }
    failed = [
        name
        for name, threshold in thresholds.items()
        if report.aggregate.get(name, 0.0) < threshold
    ]
    return len(failed) == 0, failed


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
