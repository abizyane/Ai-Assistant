from __future__ import annotations

import importlib
import pytest
from src.interface.web.api_client import RAGAPIClient

@pytest.fixture()
def client() -> RAGAPIClient:
    return RAGAPIClient(base_url="http://test", langfuse_base_url="http://langfuse")

@pytest.fixture(autouse=True)
def _reset_di_before_web_tests():
    try:
        di = importlib.import_module('src.infrastructure.di')
        if hasattr(di, '_reset_caches'):
            di._reset_caches()
    except Exception:
        pass
