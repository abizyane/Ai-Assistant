"""Unit and integration tests for DoclingLoader."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shared.errors import LoaderError

pytestmark = pytest.mark.slow


def _make_docling_document(
    *,
    markdown: str = "# Hello\n\nWorld",
    page_count: int = 2,
    has_tables: bool = False,
    section_headings: list[str] | None = None,
) -> MagicMock:
    doc = MagicMock()
    doc.export_to_markdown.return_value = markdown
    doc.pages = {i: MagicMock() for i in range(1, page_count + 1)}
    doc.tables = [MagicMock()] if has_tables else []

    if section_headings is not None:
        items = []
        for heading in section_headings:
            item = MagicMock()
            item.orig = heading
            items.append(item)
        doc.texts = items
    else:
        doc.texts = []

    return doc


def _make_conversion_result(doc: MagicMock) -> MagicMock:
    result = MagicMock()
    result.document = doc
    return result


def _make_converter(result: MagicMock) -> MagicMock:
    converter = MagicMock()
    converter.convert.return_value = result
    return converter


def _make_settings() -> Any:
    return SimpleNamespace(llm=SimpleNamespace(api_key="test"))


@pytest.fixture()
def settings() -> Any:
    return _make_settings()


@pytest.fixture()
def fake_converter() -> MagicMock:
    doc = _make_docling_document(
        markdown="# Section\n\nContent",
        page_count=3,
        has_tables=True,
        section_headings=["Section"],
    )
    return _make_converter(_make_conversion_result(doc))


async def _expect_loader_error(loader: Any, path: str) -> LoaderError:
    with pytest.raises(LoaderError) as exc:
        await loader.load(path)
    return exc.value


class TestDoclingLoaderMissing:
    def test_raises_loader_error_for_nonexistent_path(
        self, settings: Any, fake_converter: MagicMock
    ) -> None:
        with patch("docling.document_converter.DocumentConverter", return_value=fake_converter):
            from src.infrastructure.loading.docling_loader import DoclingLoader

            loader = DoclingLoader(settings)

        result = asyncio.get_event_loop().run_until_complete(
            _expect_loader_error(loader, "/tmp/nonexistent_path_xyz_12345")
        )
        assert isinstance(result, LoaderError)
        assert "does not exist" in result.message


class TestDoclingLoaderSingleFile:
    @pytest.mark.asyncio
    async def test_load_single_pdf_returns_raw_document(
        self, tmp_path: Path, settings: Any, fake_converter: MagicMock
    ) -> None:
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        with (
            patch("docling.document_converter.DocumentConverter", return_value=fake_converter),
            patch(
                "src.infrastructure.loading.docling_loader._extract_headings",
                return_value=["Section"],
            ),
        ):
            from importlib import reload

            import src.infrastructure.loading.docling_loader as mod

            reload(mod)
            loader = mod.DoclingLoader(settings)

            results = await loader.load(str(pdf_file))

        assert len(results) == 1
        raw = results[0]
        assert raw.source_path == str(pdf_file)
        assert "Section" in raw.content
        assert raw.metadata["page_count"] == 3
        assert raw.metadata["file_format"] == "pdf"
        assert raw.metadata["has_tables"] is True
        assert len(raw.content_hash) == 64


class TestDoclingLoaderDirectory:
    @pytest.mark.asyncio
    async def test_load_directory_processes_all_supported_files(
        self, tmp_path: Path, settings: Any
    ) -> None:
        (tmp_path / "doc.pdf").write_bytes(b"%PDF fake")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "page.html").write_text("<html></html>")
        (tmp_path / "data.txt").write_text("ignored")

        call_count = 0

        def fake_convert(path: str) -> MagicMock:
            nonlocal call_count
            call_count += 1
            suffix = Path(path).suffix.lstrip(".")
            doc = _make_docling_document(markdown=f"# {suffix}", page_count=1)
            return _make_conversion_result(doc)

        fake_conv = MagicMock()
        fake_conv.convert.side_effect = fake_convert

        with (
            patch("docling.document_converter.DocumentConverter", return_value=fake_conv),
            patch(
                "src.infrastructure.loading.docling_loader._extract_headings",
                return_value=[],
            ),
        ):
            from importlib import reload

            import src.infrastructure.loading.docling_loader as mod

            reload(mod)
            loader = mod.DoclingLoader(settings)
            results = await loader.load(tmp_path)

        assert call_count == 3
        assert len(results) == 3
        formats = {r.metadata["file_format"] for r in results}
        assert formats == {"pdf", "md", "html"}


class TestDoclingLoaderSkipsFailedFiles:
    @pytest.mark.asyncio
    async def test_failed_file_skipped_others_returned(self, tmp_path: Path, settings: Any) -> None:
        ok_file = tmp_path / "ok.md"
        bad_file = tmp_path / "bad.pdf"
        ok_file.write_text("# OK")
        bad_file.write_bytes(b"corrupt")

        def fake_convert(path: str) -> MagicMock:
            if "bad" in path:
                raise RuntimeError("Conversion failed")
            doc = _make_docling_document(markdown="# OK", page_count=1)
            return _make_conversion_result(doc)

        fake_conv = MagicMock()
        fake_conv.convert.side_effect = fake_convert

        with (
            patch("docling.document_converter.DocumentConverter", return_value=fake_conv),
            patch(
                "src.infrastructure.loading.docling_loader._extract_headings",
                return_value=[],
            ),
        ):
            from importlib import reload

            import src.infrastructure.loading.docling_loader as mod

            reload(mod)
            loader = mod.DoclingLoader(settings)
            results = await loader.load(tmp_path)

        assert len(results) == 1
        assert results[0].metadata["file_format"] == "md"


class TestDoclingLoaderSupports:
    def test_supports_pdf(self, tmp_path: Path, settings: Any, fake_converter: MagicMock) -> None:
        with patch("docling.document_converter.DocumentConverter", return_value=fake_converter):
            from src.infrastructure.loading.docling_loader import DoclingLoader

            loader = DoclingLoader(settings)
        assert loader.supports(str(tmp_path / "x.pdf")) is True

    def test_supports_md(self, tmp_path: Path, settings: Any, fake_converter: MagicMock) -> None:
        with patch("docling.document_converter.DocumentConverter", return_value=fake_converter):
            from src.infrastructure.loading.docling_loader import DoclingLoader

            loader = DoclingLoader(settings)
        assert loader.supports(str(tmp_path / "x.md")) is True

    def test_supports_html(self, tmp_path: Path, settings: Any, fake_converter: MagicMock) -> None:
        with patch("docling.document_converter.DocumentConverter", return_value=fake_converter):
            from src.infrastructure.loading.docling_loader import DoclingLoader

            loader = DoclingLoader(settings)
        assert loader.supports(str(tmp_path / "x.html")) is True

    def test_does_not_support_txt(
        self, tmp_path: Path, settings: Any, fake_converter: MagicMock
    ) -> None:
        with patch("docling.document_converter.DocumentConverter", return_value=fake_converter):
            from src.infrastructure.loading.docling_loader import DoclingLoader

            loader = DoclingLoader(settings)
        assert loader.supports(str(tmp_path / "x.txt")) is False

    def test_supports_directory(
        self, tmp_path: Path, settings: Any, fake_converter: MagicMock
    ) -> None:
        with patch("docling.document_converter.DocumentConverter", return_value=fake_converter):
            from src.infrastructure.loading.docling_loader import DoclingLoader

            loader = DoclingLoader(settings)
        assert loader.supports(str(tmp_path)) is True


class TestLoaderError:
    def test_loader_error_has_message(self) -> None:
        err = LoaderError("test message")
        assert err.message == "test message"
        assert err.cause is None

    def test_loader_error_with_cause(self) -> None:
        cause = ValueError("root")
        err = LoaderError("wrapped", cause=cause)
        assert err.cause is cause

    def test_loader_error_is_exception(self) -> None:
        err = LoaderError("boom")
        assert isinstance(err, Exception)
