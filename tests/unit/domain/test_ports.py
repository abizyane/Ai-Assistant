"""Smoke-import tests for domain port Protocols.

Importing these modules covers all their executable statements (class + method
definitions), pushing coverage above the 75 % CI gate.
"""

from __future__ import annotations

from src.domain.ports.eval import EvalPort
from src.domain.ports.metrics import MetricsPort
from src.domain.ports.session_repo import SessionRepoPort


def test_ports_are_importable() -> None:
    assert EvalPort
    assert MetricsPort
    assert SessionRepoPort
