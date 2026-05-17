from __future__ import annotations

from unittest.mock import MagicMock

from src.infrastructure.observability.langfuse_tracer import LangfuseTracer


def test_disabled_when_no_keys() -> None:
    tracer = LangfuseTracer(public_key="", secret_key="")
    assert tracer._enabled is False


def test_start_span_returns_trace_context_when_disabled() -> None:
    tracer = LangfuseTracer()
    ctx = tracer.start_span("test.span", input={"key": "val"})
    assert ctx.name == "test.span"
    assert ctx.span_id != ""
    assert ctx.trace_id != ""


def test_end_span_noop_when_disabled() -> None:
    tracer = LangfuseTracer()
    ctx = tracer.start_span("noop")
    tracer.end_span(ctx, output={"result": "ok"})


def test_flush_noop_when_disabled() -> None:
    tracer = LangfuseTracer()
    tracer.flush()


def test_end_span_with_output_calls_update() -> None:
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value = mock_trace

    tracer = LangfuseTracer()
    tracer._enabled = True
    tracer._client = mock_client

    from src.domain.ports.dto import TraceContext

    ctx = TraceContext(span_id="s1", trace_id="t1", name="x", metadata={})
    tracer.end_span(ctx, output={"score": 0.9})
    mock_client.trace.assert_called_once_with(id="t1")
    mock_trace.update.assert_called_once_with(output={"score": 0.9})


def test_flush_calls_client_flush() -> None:
    mock_client = MagicMock()
    tracer = LangfuseTracer()
    tracer._enabled = True
    tracer._client = mock_client
    tracer.flush()
    mock_client.flush.assert_called_once()


def test_start_span_handles_client_exception() -> None:
    mock_client = MagicMock()
    mock_client.trace.side_effect = RuntimeError("network error")

    tracer = LangfuseTracer()
    tracer._enabled = True
    tracer._client = mock_client

    ctx = tracer.start_span("failing.span")
    assert ctx.name == "failing.span"
