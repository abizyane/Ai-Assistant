from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.use_cases.evaluate import EvaluateUseCase, check_thresholds
from src.domain.entities.answer import AnswerWithCitations
from src.domain.entities.evaluation import EvaluationReport

_SMOKE_FIXTURE = Path("tests/fixtures/eval_smoke.jsonl")

_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

_FAKE_SCORES: dict[str, list[float]] = {
    "faithfulness": [0.90, 0.80, 0.70, 0.85, 0.95],
    "answer_relevancy": [0.80, 0.75, 0.82, 0.78, 0.90],
    "context_precision": [0.70, 0.65, 0.72, 0.68, 0.80],
    "context_recall": [0.60, 0.55, 0.62, 0.58, 0.70],
    "answer_correctness": [0.75, 0.70, 0.72, 0.78, 0.80],
}


def _make_mock_result() -> MagicMock:
    mock_result = MagicMock()
    mock_result.__getitem__.side_effect = lambda key: _FAKE_SCORES[key]
    return mock_result


def _make_runner(text: str = "Mock answer") -> AsyncMock:
    return AsyncMock(
        return_value=AnswerWithCitations(
            text=text,
            citations=[],
            language="en",
            tokens_in=0,
            tokens_out=0,
        )
    )


def _make_settings(
    faithfulness: float = 0.85,
    answer_relevancy: float = 0.80,
    context_precision: float = 0.75,
    context_recall: float = 0.70,
    answer_correctness: float = 0.70,
) -> MagicMock:
    settings = MagicMock()
    settings.eval.faithfulness = faithfulness
    settings.eval.answer_relevancy = answer_relevancy
    settings.eval.context_precision = context_precision
    settings.eval.context_recall = context_recall
    settings.eval.answer_correctness = answer_correctness
    return settings


@pytest.mark.slow
async def test_happy_path_5_rows(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    with patch("src.application.use_cases.evaluate.evaluate") as mock_eval:
        mock_eval.return_value = _make_mock_result()
        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, output_dir=tmp_path)

    assert report.sample_size == 5
    assert len(report.per_question) == 5
    assert runner.call_count == 5
    assert set(report.aggregate.keys()) == set(_METRIC_NAMES)
    for pq in report.per_question:
        assert "query" in pq
        assert "prediction" in pq
        assert "ground_truth" in pq
        assert "contexts" in pq
        for name in _METRIC_NAMES:
            assert name in pq


@pytest.mark.slow
async def test_sample_limits_rows(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    fake_scores_2: dict[str, list[float]] = {k: v[:2] for k, v in _FAKE_SCORES.items()}
    mock_result = MagicMock()
    mock_result.__getitem__.side_effect = lambda key: fake_scores_2[key]

    with patch("src.application.use_cases.evaluate.evaluate") as mock_eval:
        mock_eval.return_value = mock_result
        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, sample=2, output_dir=tmp_path)

    assert report.sample_size == 2
    assert len(report.per_question) == 2
    assert runner.call_count == 2


async def test_empty_dataset_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    runner = _make_runner()
    settings = _make_settings()
    uc = EvaluateUseCase(agent_runner=runner, settings=settings)

    with pytest.raises(ValueError, match="Empty dataset"):
        await uc.execute(empty, output_dir=tmp_path)


async def test_check_thresholds_all_pass() -> None:
    report = EvaluationReport(
        timestamp=datetime.now(UTC),
        dataset_path="tests/fixtures/eval_smoke.jsonl",
        sample_size=5,
        per_question=[],
        aggregate={
            "faithfulness": 0.90,
            "answer_relevancy": 0.85,
            "context_precision": 0.80,
            "context_recall": 0.75,
            "answer_correctness": 0.75,
        },
    )
    settings = _make_settings()

    passed, failed = check_thresholds(report, settings)

    assert passed is True
    assert failed == []


async def test_check_thresholds_faithfulness_fails() -> None:
    report = EvaluationReport(
        timestamp=datetime.now(UTC),
        dataset_path="tests/fixtures/eval_smoke.jsonl",
        sample_size=5,
        per_question=[],
        aggregate={
            "faithfulness": 0.50,
            "answer_relevancy": 0.85,
            "context_precision": 0.80,
            "context_recall": 0.75,
            "answer_correctness": 0.75,
        },
    )
    settings = _make_settings()

    passed, failed = check_thresholds(report, settings)

    assert passed is False
    assert "faithfulness" in failed
    assert len(failed) == 1


@pytest.mark.slow
async def test_aggregate_scores_are_means(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    with patch("src.application.use_cases.evaluate.evaluate") as mock_eval:
        mock_eval.return_value = _make_mock_result()
        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, output_dir=tmp_path)

    expected_faith = sum(_FAKE_SCORES["faithfulness"]) / len(_FAKE_SCORES["faithfulness"])
    assert report.aggregate["faithfulness"] == pytest.approx(expected_faith)


@pytest.mark.slow
async def test_report_written_to_output_dir(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    with patch("src.application.use_cases.evaluate.evaluate") as mock_eval:
        mock_eval.return_value = _make_mock_result()
        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, output_dir=tmp_path)

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    content = json_files[0].read_text(encoding="utf-8")
    assert report.dataset_path in content
