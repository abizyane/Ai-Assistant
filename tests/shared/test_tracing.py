from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace.status import StatusCode

from src.shared.tracing import (
    current_session_id,
    current_trace_id,
    traced,
)


@pytest.fixture(autouse=True)
def _reset_contextvars() -> None:
    current_trace_id.set(None)
    current_session_id.set(None)


@pytest.fixture()
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    test_tracer = provider.get_tracer("test")
    monkeypatch.setattr("src.shared.tracing.get_tracer", lambda _: test_tracer)
    return exp


async def test_span_created_with_correct_name(exporter: InMemorySpanExporter) -> None:
    @traced("test.operation")
    async def my_func() -> str:
        return "hello"

    result = await my_func()

    assert result == "hello"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test.operation"


async def test_default_attrs_set_on_span(exporter: InMemorySpanExporter) -> None:
    @traced("test.attrs", key="val", num=42)
    async def my_func() -> None:
        return None

    await my_func()

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs is not None
    assert attrs.get("key") == "val"
    assert attrs.get("num") == 42


async def test_exception_recorded_and_status_error(exporter: InMemorySpanExporter) -> None:
    @traced("test.error")
    async def failing_func() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await failing_func()

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1


async def test_exception_reraised(exporter: InMemorySpanExporter) -> None:
    @traced("test.reraise")
    async def failing_func() -> None:
        raise RuntimeError("propagated")

    with pytest.raises(RuntimeError, match="propagated"):
        await failing_func()


async def test_contextvars_propagate_across_await(exporter: InMemorySpanExporter) -> None:
    current_trace_id.set("trace-abc")
    current_session_id.set("sess-xyz")

    @traced("test.ctx")
    async def my_func() -> str:
        return "ok"

    await my_func()

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs is not None
    assert attrs.get("trace.id") == "trace-abc"
    assert attrs.get("session.id") == "sess-xyz"


async def test_only_non_none_contextvars_set(exporter: InMemorySpanExporter) -> None:
    current_trace_id.set(None)
    current_session_id.set("sess-only")

    @traced("test.partial_ctx")
    async def my_func() -> None:
        return None

    await my_func()

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs is not None
    assert "trace.id" not in attrs
    assert attrs.get("session.id") == "sess-only"


async def test_gen_ai_usage_extracted_from_result(exporter: InMemorySpanExporter) -> None:
    class _FakeResult:
        input_tokens = 10
        output_tokens = 20

    @traced("test.genai")
    async def my_func() -> _FakeResult:
        return _FakeResult()

    await my_func()

    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs is not None
    assert attrs.get("gen_ai.usage.input_tokens") == 10
    assert attrs.get("gen_ai.usage.output_tokens") == 20


async def test_no_gen_ai_attrs_when_result_lacks_tokens(exporter: InMemorySpanExporter) -> None:
    @traced("test.no_genai")
    async def my_func() -> str:
        return "plain string"

    await my_func()

    attrs = exporter.get_finished_spans()[0].attributes or {}
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.usage.output_tokens" not in attrs


async def test_non_async_function_returned_unchanged(exporter: InMemorySpanExporter) -> None:
    @traced("test.sync")
    def sync_func() -> int:  # type: ignore[misc]
        return 42

    result = sync_func()
    assert result == 42
    assert len(exporter.get_finished_spans()) == 0


async def test_return_value_preserved(exporter: InMemorySpanExporter) -> None:
    @traced("test.return")
    async def my_func() -> list[int]:
        return [1, 2, 3]

    result = await my_func()
    assert result == [1, 2, 3]


async def test_span_finished_even_on_success(exporter: InMemorySpanExporter) -> None:
    @traced("test.finish")
    async def my_func() -> None:
        return None

    await my_func()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1


async def test_contextvars_independent_across_calls(exporter: InMemorySpanExporter) -> None:
    current_session_id.set("sess-1")

    @traced("test.multi")
    async def my_func() -> str:
        return "x"

    await my_func()
    current_session_id.set("sess-2")
    await my_func()

    spans = exporter.get_finished_spans()
    assert spans[0].attributes is not None
    assert spans[1].attributes is not None
    assert spans[0].attributes.get("session.id") == "sess-1"
    assert spans[1].attributes.get("session.id") == "sess-2"
