"""Prometheus metrics helpers: REGISTRY, counters, histograms, gauges, decorators."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    make_asgi_app,
)

# Single shared registry — importable by adapters
REGISTRY = CollectorRegistry()

# Pre-declared core metrics (avoid duplicate registration errors in tests)
_COUNTERS: dict[str, Counter] = {}
_HISTOGRAMS: dict[str, Histogram] = {}
_GAUGES: dict[str, Gauge] = {}


def _get_or_create_counter(name: str, labels: list[str] | None = None) -> Counter:
    if name not in _COUNTERS:
        _COUNTERS[name] = Counter(name, name.replace("_", " "), labels or [], registry=REGISTRY)
    return _COUNTERS[name]


def _get_or_create_histogram(name: str, labels: list[str] | None = None) -> Histogram:
    if name not in _HISTOGRAMS:
        _HISTOGRAMS[name] = Histogram(name, name.replace("_", " "), labels or [], registry=REGISTRY)
    return _HISTOGRAMS[name]


def _get_or_create_gauge(name: str, labels: list[str] | None = None) -> Gauge:
    if name not in _GAUGES:
        _GAUGES[name] = Gauge(name, name.replace("_", " "), labels or [], registry=REGISTRY)
    return _GAUGES[name]


def inc_counter(
    name: str,
    labels: dict[str, str] | None = None,
    amount: int = 1,
) -> None:
    """Increment a counter metric by *amount* (default 1)."""
    c = _get_or_create_counter(name, list(labels.keys()) if labels else None)
    if labels:
        c.labels(**labels).inc(amount)
    else:
        c.inc(amount)


def observe_histogram(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    """Record a value in a histogram metric."""
    h = _get_or_create_histogram(name, list(labels.keys()) if labels else None)
    if labels:
        h.labels(**labels).observe(value)
    else:
        h.observe(value)


def set_gauge(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    """Set a gauge metric to a specific value."""
    g = _get_or_create_gauge(name, list(labels.keys()) if labels else None)
    if labels:
        g.labels(**labels).set(value)
    else:
        g.set(value)


F = TypeVar("F", bound=Callable[..., Any])


def track_latency(metric_name: str, labels: dict[str, str] | None = None) -> Callable[[F], F]:
    """Decorator: records function execution time into a histogram."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                observe_histogram(metric_name, time.perf_counter() - start, labels)

        return wrapper  # type: ignore[return-value] — wrapper preserves callable signature via functools.wraps

    return decorator


def metrics_asgi_app() -> Any:  # noqa: ANN401
    """Return a prometheus_client ASGI app for mounting on /metrics."""
    return make_asgi_app(registry=REGISTRY)


def get_metrics_output() -> str:
    """Return current metrics as Prometheus text format (for testing)."""
    return generate_latest(REGISTRY).decode()


# ---------------------------------------------------------------------------
# Pre-registered metrics (avoid duplicate registration on first request)
# ---------------------------------------------------------------------------

# HTTP request duration histogram — consumed by RequestLoggingMiddleware
_get_or_create_histogram(
    "http_request_duration_seconds",
    ["method", "path", "status"],
)
