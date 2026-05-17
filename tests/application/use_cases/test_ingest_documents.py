from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.use_cases.ingest_documents import IngestDocumentsUseCase, _detect_language
from src.domain.ports.dto import ChunkContent, IngestionReport, RawDocument


def _make_raw_doc(
    content: str = "Hello world text",
    content_hash: str = "abc123",
    source_path: str = "/fake/doc.pdf",
) -> RawDocument:
    return RawDocument(
        source_path=source_path,
        content=content,
        language="en",
        content_hash=content_hash,
    )


def _make_chunk(content: str = "Chunk text", position: int = 0) -> ChunkContent:
    return ChunkContent(content=content, position=position, token_count=len(content.split()))


def _make_session_factory(return_none: bool = True) -> MagicMock:
    mock_session = AsyncMock()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=mock_cm)

    result = MagicMock()
    result.scalar_one_or_none.return_value = None if return_none else MagicMock()
    mock_session.execute = AsyncMock(return_value=result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    return factory


@pytest.fixture()
def loader() -> MagicMock:
    m = MagicMock()
    m.load = AsyncMock(return_value=[_make_raw_doc()])
    m.supports = MagicMock(return_value=True)
    return m


@pytest.fixture()
def chunker() -> MagicMock:
    m = MagicMock()
    m.chunk = MagicMock(return_value=[_make_chunk("First chunk"), _make_chunk("Second chunk", 1)])
    return m


@pytest.fixture()
def embedder() -> MagicMock:
    m = MagicMock()
    m.embed_texts = AsyncMock(return_value=[[0.1] * 1024, [0.2] * 1024])
    return m


@pytest.fixture()
def vector_store() -> MagicMock:
    m = MagicMock()
    m.upsert = AsyncMock(return_value=2)
    return m


@pytest.fixture()
def session_factory() -> MagicMock:
    return _make_session_factory(return_none=True)


@pytest.fixture()
def tracer() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def logger() -> logging.Logger:
    return MagicMock(spec=logging.Logger)


@pytest.fixture()
def use_case(
    loader: MagicMock,
    chunker: MagicMock,
    embedder: MagicMock,
    vector_store: MagicMock,
    session_factory: MagicMock,
    tracer: MagicMock,
    logger: logging.Logger,
) -> IngestDocumentsUseCase:
    return IngestDocumentsUseCase(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        session_repo=session_factory,
        tracer=tracer,
        logger=logger,
    )


async def test_empty_path_returns_empty_report(
    use_case: IngestDocumentsUseCase,
    tmp_path: Path,
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    report = await use_case.execute(empty_dir)

    assert isinstance(report, IngestionReport)
    assert report.files_processed == 0
    assert report.files_skipped == 0
    assert report.chunks_created == 0
    assert report.errors == []


async def test_nonexistent_path_returns_empty_report(
    use_case: IngestDocumentsUseCase,
    tmp_path: Path,
) -> None:
    report = await use_case.execute(tmp_path / "does_not_exist")

    assert report.files_processed == 0
    assert report.files_skipped == 0
    assert report.chunks_created == 0


async def test_single_file_processed(
    use_case: IngestDocumentsUseCase,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf content")

    report = await use_case.execute(pdf)

    assert report.files_processed == 1
    assert report.files_skipped == 0
    assert report.chunks_created == 2
    assert report.errors == []
    assert report.duration_seconds >= 0.0


async def test_idempotency_skips_already_processed_file(
    loader: MagicMock,
    chunker: MagicMock,
    embedder: MagicMock,
    vector_store: MagicMock,
    tracer: MagicMock,
    logger: logging.Logger,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"stable content")

    already_exists_factory = _make_session_factory(return_none=False)

    uc = IngestDocumentsUseCase(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        session_repo=already_exists_factory,
        tracer=tracer,
        logger=logger,
    )

    report = await uc.execute(pdf)

    assert report.files_processed == 0
    assert report.files_skipped == 1
    assert report.chunks_created == 0
    loader.load.assert_not_called()
    embedder.embed_texts.assert_not_called()


async def test_language_detection_per_chunk(
    use_case: IngestDocumentsUseCase,
    chunker: MagicMock,
    tmp_path: Path,
) -> None:
    chunker.chunk.return_value = [
        _make_chunk("This is English text.", 0),
        _make_chunk("Ceci est du texte français.", 1),
    ]

    pdf = tmp_path / "multilang.pdf"
    pdf.write_bytes(b"multilingual content")

    with patch("src.application.use_cases.ingest_documents._detect_language") as mock_detect:
        mock_detect.side_effect = ["en", "fr"]
        report = await use_case.execute(pdf)

    assert report.chunks_created == 2
    assert mock_detect.call_count == 2


async def test_language_hint_overrides_detection(
    use_case: IngestDocumentsUseCase,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"some content")

    with patch("src.application.use_cases.ingest_documents._detect_language") as mock_detect:
        report = await use_case.execute(pdf, language_hint="ar")

    mock_detect.assert_not_called()
    assert report.files_processed == 1


async def test_error_aggregation_continues_on_corrupt_file(
    loader: MagicMock,
    chunker: MagicMock,
    embedder: MagicMock,
    vector_store: MagicMock,
    session_factory: MagicMock,
    tracer: MagicMock,
    logger: logging.Logger,
    tmp_path: Path,
) -> None:
    good_pdf = tmp_path / "good.pdf"
    good_pdf.write_bytes(b"good content")
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"bad content")

    call_count = 0

    async def load_side_effect(path: str) -> list[RawDocument]:
        nonlocal call_count
        call_count += 1
        if "bad" in path:
            raise ValueError("corrupt file")
        return [_make_raw_doc()]

    loader.load.side_effect = load_side_effect

    uc = IngestDocumentsUseCase(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        session_repo=session_factory,
        tracer=tracer,
        logger=logger,
    )

    report = await uc.execute(tmp_path)

    assert report.files_processed == 1
    assert len(report.errors) == 1
    assert "corrupt file" in report.errors[0][1]


async def test_metrics_emitted(
    use_case: IngestDocumentsUseCase,
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"content for metrics test")

    with (
        patch("src.application.use_cases.ingest_documents.inc_counter") as mock_inc,
        patch("src.application.use_cases.ingest_documents.observe_histogram") as mock_obs,
    ):
        report = await use_case.execute(pdf)

    assert mock_inc.called
    file_calls = [c for c in mock_inc.call_args_list if c.args[0] == "ingestion_files_total"]
    chunk_calls = [c for c in mock_inc.call_args_list if c.args[0] == "ingestion_chunks_total"]
    assert len(file_calls) >= 1
    assert len(chunk_calls) == report.chunks_created

    assert mock_obs.called
    hist_calls = [c for c in mock_obs.call_args_list if c.args[0] == "ingestion_duration_seconds"]
    assert len(hist_calls) == 1


async def test_directory_ingests_all_pdfs(
    use_case: IngestDocumentsUseCase,
    loader: MagicMock,
    tmp_path: Path,
) -> None:
    for i in range(3):
        (tmp_path / f"doc{i}.pdf").write_bytes(f"content {i}".encode())

    loader.load.side_effect = lambda path: [_make_raw_doc(content_hash=path[-5:])]

    report = await use_case.execute(tmp_path)

    assert report.files_processed == 3
    assert report.chunks_created == 6


def test_detect_language_returns_unknown_on_failure() -> None:
    with patch("src.application.use_cases.ingest_documents.detect") as mock_detect:
        from langdetect import LangDetectException
        mock_detect.side_effect = LangDetectException(0, "fail")
        result = _detect_language("!!!###")

    assert result == "unknown"


def test_detect_language_returns_code() -> None:
    with patch("src.application.use_cases.ingest_documents.detect", return_value="en"):
        result = _detect_language("This is English")
    assert result == "en"
