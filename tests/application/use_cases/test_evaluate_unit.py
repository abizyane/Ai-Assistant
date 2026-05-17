from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.use_cases.evaluate import (
    EvaluateUseCase,
    default_agent_runner,
)
from src.domain.entities.answer import AnswerWithCitations

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


def _make_runner() -> AsyncMock:
    runner = AsyncMock()
    runner.return_value = AnswerWithCitations(
        text="Test answer",
        citations=[],
        language="en",
        tokens_in=10,
        tokens_out=5,
    )
    return runner


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.eval.faithfulness = 0.80
    settings.eval.answer_relevancy = 0.75
    settings.eval.context_precision = 0.65
    settings.eval.context_recall = 0.55
    settings.eval.answer_correctness = 0.70
    return settings


def _make_mock_result() -> MagicMock:
    mock_result = MagicMock()
    mock_result.__getitem__.side_effect = lambda key: _FAKE_SCORES[key]
    return mock_result


async def test_execute_with_mocked_dataset(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    mock_dataset = MagicMock()

    with (
        patch("src.application.use_cases.evaluate.Dataset") as mock_ds_cls,
        patch("src.application.use_cases.evaluate.evaluate") as mock_eval,
    ):
        mock_ds_cls.from_dict.return_value = mock_dataset
        mock_eval.return_value = _make_mock_result()

        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, output_dir=tmp_path)

    assert report.sample_size == 5
    assert len(report.per_question) == 5
    assert set(report.aggregate.keys()) == set(_METRIC_NAMES)


async def test_execute_sample_limits_rows(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    fake_scores_2 = {k: v[:2] for k, v in _FAKE_SCORES.items()}
    mock_result = MagicMock()
    mock_result.__getitem__.side_effect = lambda key: fake_scores_2[key]

    with (
        patch("src.application.use_cases.evaluate.Dataset") as mock_ds_cls,
        patch("src.application.use_cases.evaluate.evaluate") as mock_eval,
    ):
        mock_ds_cls.from_dict.return_value = MagicMock()
        mock_eval.return_value = mock_result

        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, sample=2, output_dir=tmp_path)

    assert report.sample_size == 2
    assert runner.call_count == 2


async def test_execute_aggregate_scores_are_means(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    with (
        patch("src.application.use_cases.evaluate.Dataset") as mock_ds_cls,
        patch("src.application.use_cases.evaluate.evaluate") as mock_eval,
    ):
        mock_ds_cls.from_dict.return_value = MagicMock()
        mock_eval.return_value = _make_mock_result()

        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, output_dir=tmp_path)

    expected = sum(_FAKE_SCORES["faithfulness"]) / len(_FAKE_SCORES["faithfulness"])
    assert report.aggregate["faithfulness"] == pytest.approx(expected)


async def test_execute_report_written_to_output_dir(tmp_path: Path) -> None:
    runner = _make_runner()
    settings = _make_settings()

    with (
        patch("src.application.use_cases.evaluate.Dataset") as mock_ds_cls,
        patch("src.application.use_cases.evaluate.evaluate") as mock_eval,
    ):
        mock_ds_cls.from_dict.return_value = MagicMock()
        mock_eval.return_value = _make_mock_result()

        uc = EvaluateUseCase(agent_runner=runner, settings=settings)
        report = await uc.execute(_SMOKE_FIXTURE, output_dir=tmp_path)

    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    content = json_files[0].read_text(encoding="utf-8")
    assert report.dataset_path in content


async def test_default_agent_runner_returns_answer() -> None:
    mock_graph = AsyncMock()
    mock_answer = AnswerWithCitations(
        text="answer", citations=[], language="en", tokens_in=1, tokens_out=1
    )
    mock_graph.ainvoke.return_value = {"final_answer": mock_answer}

    runner = default_agent_runner(mock_graph)
    result = await runner("What is 1337?", "en")
    assert result.text == "answer"


async def test_default_agent_runner_no_answer_returns_fallback() -> None:
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"final_answer": None}

    runner = default_agent_runner(mock_graph)
    result = await runner("What is 1337?", "en")
    assert result.text == "No answer generated."
