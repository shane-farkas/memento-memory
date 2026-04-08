"""Shared test fixtures for Memento tests."""

import pytest

from memento.config import MementoConfig
from memento.db import Database


class FakeLLMClient:
    """Fake LLM client for testing. Returns configurable canned responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0

    def complete(
        self,
        messages: list[dict],
        model: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self._responses:
            idx = min(self._call_count, len(self._responses) - 1)
            self._call_count += 1
            return self._responses[idx]
        return "[]"


@pytest.fixture
def db():
    """Provide an in-memory SQLite database for tests."""
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def config():
    """Provide a default MementoConfig for tests."""
    return MementoConfig(db_path=":memory:")
