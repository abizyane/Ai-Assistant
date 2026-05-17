"""Metrics port — contract for application metrics emission adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsPort(Protocol):
    """Protocol for metrics emission adapters used by the DI container."""

    def inc_counter(self, name: str, labels: dict[str, str] | None = None) -> None:
        """Increment a named counter metric by one, optionally with labels."""
        ...

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a named histogram observation, optionally with labels."""
        ...

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a named gauge metric to the given value, optionally with labels."""
        ...
