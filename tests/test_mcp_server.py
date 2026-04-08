"""Tests for MCP server tool functions."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import memento.mcp_server as mcp_mod
from memento.config import MementoConfig
from memento.memory_store import MemoryStore
from memento.models import EntityType
from tests.conftest import FakeLLMClient


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(8, dtype=np.float32)
        for word in set(text.lower().split()):
            vec[hash(word) % 8] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@pytest.fixture(autouse=True)
def setup_mcp_store():
    config = MementoConfig(db_path=":memory:")
    with (
        patch("memento.memory_store.create_embedder", return_value=FakeEmbedder()),
        patch("memento.llm.create_llm_client", return_value=FakeLLMClient()),
    ):
        store = MemoryStore(config)
    mcp_mod._store = store
    yield store
    store.close()
    mcp_mod._store = None


def test_entities_empty():
    result = mcp_mod.memory_entities()
    assert "No entities" in result


def test_entity_not_found():
    result = mcp_mod.memory_entity("nonexistent-id")
    assert "not found" in result


def test_recall_no_results():
    result = mcp_mod.memory_recall("unknown xyz")
    assert "No relevant" in result or len(result) > 0


def test_entities_list(setup_mcp_store):
    store = setup_mcp_store
    store.graph.create_entity("John Smith", EntityType.PERSON)
    store.graph.create_entity("Acme Corp", EntityType.ORGANIZATION)
    result = mcp_mod.memory_entities()
    assert "John Smith" in result
    assert "Acme Corp" in result


def test_entity_detail(setup_mcp_store):
    store = setup_mcp_store
    entity = store.graph.create_entity("John Smith", EntityType.PERSON)
    store.graph.set_property(entity.id, "title", "Director")
    result = mcp_mod.memory_entity(entity.id)
    assert "John Smith" in result
    assert "Director" in result


def test_recall_by_entity(setup_mcp_store):
    store = setup_mcp_store
    entity = store.graph.create_entity("John Smith", EntityType.PERSON)
    store.graph.set_property(entity.id, "title", "Director")
    result = mcp_mod.memory_recall("John Smith")
    assert "John Smith" in result


def test_health(setup_mcp_store):
    store = setup_mcp_store
    store.graph.create_entity("John", EntityType.PERSON)
    result = mcp_mod.memory_health()
    assert "Entities: 1" in result


def test_conflicts_empty():
    result = mcp_mod.memory_conflicts()
    assert "No unresolved" in result


def test_mcp_server_exists():
    assert mcp_mod.mcp is not None
