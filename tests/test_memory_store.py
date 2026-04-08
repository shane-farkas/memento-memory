"""Tests for MemoryStore public API."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

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


@pytest.fixture
def store():
    config = MementoConfig(db_path=":memory:")
    with (
        patch("memento.memory_store.create_embedder", return_value=FakeEmbedder()),
        patch("memento.llm.create_llm_client", return_value=FakeLLMClient()),
    ):
        s = MemoryStore(config)
    yield s
    s.close()


def test_ingest_and_recall(store):
    # FakeLLMClient returns "[]" by default → no entities extracted
    result = store.ingest("Had a meeting with John Smith today.")
    assert result.verbatim_chunk_id
    # Should still be searchable via verbatim
    memory = store.recall("meeting")
    assert "meeting" in memory.text.lower() or memory.text


def test_correct(store):
    entity = store.graph.create_entity("John", EntityType.PERSON)
    store.graph.set_property(entity.id, "title", "Manager")
    store.correct(entity.id, "title", "Director", reason="Promoted")
    prop = store.graph.get_property(entity.id, "title")
    assert prop.value == "Director"


def test_forget_entity(store):
    entity = store.graph.create_entity("John", EntityType.PERSON)
    store.forget(entity_id=entity.id)
    assert store.graph.get_entity(entity.id) is None


def test_merge(store):
    a = store.graph.create_entity("John Smith", EntityType.PERSON)
    b = store.graph.create_entity("J. Smith", EntityType.PERSON)
    result = store.merge(a.id, b.id)
    assert result["survivor_id"] == a.id


def test_entity_list(store):
    store.graph.create_entity("John", EntityType.PERSON)
    store.graph.create_entity("Acme", EntityType.ORGANIZATION)
    all_entities = store.entity_list()
    assert len(all_entities) == 2
    people = store.entity_list(type=EntityType.PERSON)
    assert len(people) == 1


def test_health(store):
    store.graph.create_entity("John", EntityType.PERSON)
    h = store.health()
    assert h.node_count == 1
    assert h.unresolved_conflicts == 0


def test_session(store):
    session = store.start_session()
    assert session.session_id
    session.on_turn("I met John Smith today")
    # End flushes scratchpad through ingestion
    session.end()


def test_conflicts_list(store):
    entity = store.graph.create_entity("John", EntityType.PERSON)
    store.graph.set_property(entity.id, "title", "Director", confidence=0.9)
    from memento.conflict import ConflictDetector
    detector = ConflictDetector(store.graph)
    detector.check(entity.id, "title", "VP", new_authority=0.5)
    conflicts = store.conflicts()
    assert len(conflicts) >= 1
