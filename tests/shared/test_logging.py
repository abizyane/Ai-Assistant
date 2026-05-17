from __future__ import annotations

import logging

from src.shared.logging import bind_context, clear_context, configure_logging, get_logger


def test_configure_logging_sets_root_handler() -> None:
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert len(root.handlers) >= 1
    assert root.level == logging.DEBUG


def test_configure_logging_info_level() -> None:
    configure_logging("INFO")
    root = logging.getLogger()
    assert root.level == logging.INFO


def test_get_logger_returns_bound_logger() -> None:
    logger = get_logger("test.module")
    assert logger is not None


def test_bind_and_clear_context() -> None:
    bind_context(trace_id="abc123", session_id="sess456")
    clear_context()


def test_bind_context_empty_strings() -> None:
    bind_context()
    clear_context()
