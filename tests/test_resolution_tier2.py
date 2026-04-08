"""Tests for entity resolution tier 2 (full multi-signal + LLM tiebreaker)."""

from __future__ import annotations

import numpy as np
import pytest

from memento.db import Database
from memento.entity_resolution import Tier2EntityResolver, Tier2Signals
from memento.extraction import ExtractedEntity
from memento.graph_store import GraphStore
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


@pytest.fixture
def resolver():
    db = Database(":memory:")
    graph = GraphStore(db)
    llm = FakeLLMClient(responses=["NO"])
    r = Tier2EntityResolver(graph, embedder=FakeEmbedder(), llm_client=llm)
    yield r, graph
    db.close()


@pytest.fixture
def resolver_no_embedder():
    db = Database(":memory:")
    graph = GraphStore(db)
    llm = FakeLLMClient(responses=["NO"])
    r = Tier2EntityResolver(graph, embedder=None, llm_client=llm)
    yield r, graph
    db.close()


def test_tier2_exact_match(resolver):
    r, graph = resolver
    existing = graph.create_entity("John Smith", EntityType.PERSON)
    result = r.resolve(ExtractedEntity("John Smith", EntityType.PERSON))
    assert result.action == "merge"
    assert result.entity.id == existing.id


def test_tier2_no_match(resolver):
    r, graph = resolver
    graph.create_entity("Alice Brown", EntityType.PERSON)
    result = r.resolve(ExtractedEntity("Bob Johnson", EntityType.PERSON))
    assert result.action == "create"


def test_tier2_type_disqualifier(resolver):
    r, graph = resolver
    graph.create_entity("John's Pizza", EntityType.ORGANIZATION)
    result = r.resolve(ExtractedEntity("John", EntityType.PERSON))
    assert result.action == "create"


def test_tier2_scoring_includes_embeddings(resolver):
    r, graph = resolver
    graph.create_entity("John Smith", EntityType.PERSON)
    signals = r._score_candidate_tier2(
        ExtractedEntity("Jon Smith", EntityType.PERSON),
        graph.search_entities(name="John Smith")[0],
    )
    assert isinstance(signals, Tier2Signals)
    assert signals.description_sim is not None


def test_tier2_works_without_embedder(resolver_no_embedder):
    r, graph = resolver_no_embedder
    graph.create_entity("John Smith", EntityType.PERSON)
    result = r.resolve(ExtractedEntity("John Smith", EntityType.PERSON))
    assert result.action == "merge"


def test_tier2_shared_neighbors_signal(resolver):
    r, graph = resolver
    john = graph.create_entity("John Smith", EntityType.PERSON)
    acme = graph.create_entity("Acme Corp", EntityType.ORGANIZATION)
    graph.create_relationship(john.id, acme.id, "works_at")
    signals = r._score_candidate_tier2(ExtractedEntity("Jon Smith", EntityType.PERSON), john)
    assert signals.shared_neighbors is not None
    assert signals.shared_neighbors > 0.0


def test_cold_start_sparse_graph(resolver):
    r, graph = resolver
    graph.create_entity("A", EntityType.PERSON)
    graph.create_entity("B", EntityType.PERSON)
    weights = r._get_adaptive_weights()
    assert weights["shared_neighbors"] < 0.05


def test_cold_start_dense_graph(resolver):
    r, graph = resolver
    entities = [graph.create_entity(f"Entity {i}", EntityType.PERSON) for i in range(10)]
    for i in range(len(entities)):
        for j in range(i + 1, min(i + 4, len(entities))):
            graph.create_relationship(entities[i].id, entities[j].id, "knows")
    weights = r._get_adaptive_weights()
    assert weights["shared_neighbors"] > 0.05


def test_recency_signal(resolver):
    r, graph = resolver
    graph.create_entity("John Smith", EntityType.PERSON)
    signals = r._score_candidate_tier2(
        ExtractedEntity("John Smith", EntityType.PERSON),
        graph.search_entities(name="John Smith")[0],
    )
    assert signals.recency is not None
    assert signals.recency > 0.5


def test_llm_tiebreaker_yes(resolver):
    r, graph = resolver
    r._llm = FakeLLMClient(responses=["YES"])
    existing = graph.create_entity("John Smith", EntityType.PERSON)
    result = r._llm_tiebreaker(
        ExtractedEntity("J. Smith", EntityType.PERSON), existing
    )
    assert result is True


def test_llm_tiebreaker_no(resolver):
    r, graph = resolver
    r._llm = FakeLLMClient(responses=["NO"])
    existing = graph.create_entity("John Smith", EntityType.PERSON)
    result = r._llm_tiebreaker(
        ExtractedEntity("Jane Smith", EntityType.PERSON), existing
    )
    assert result is False


def test_llm_tiebreaker_error(resolver):
    r, _ = resolver

    class ErrorLLM:
        def complete(self, **kwargs):
            raise Exception("API error")

    r._llm = ErrorLLM()
    from memento.models import Entity
    candidate = Entity(name="John Smith", type=EntityType.PERSON)
    assert r._llm_tiebreaker(ExtractedEntity("John", EntityType.PERSON), candidate) is False


def test_llm_tiebreaker_none():
    """No LLM client → always returns False."""
    db = Database(":memory:")
    graph = GraphStore(db)
    r = Tier2EntityResolver(graph, llm_client=None)
    from memento.models import Entity
    candidate = Entity(name="John", type=EntityType.PERSON)
    assert r._llm_tiebreaker(ExtractedEntity("John", EntityType.PERSON), candidate) is False
    db.close()


def test_disqualifier_type_incompatible(resolver):
    r, _ = resolver
    from memento.models import Entity
    candidate = Entity(name="John's Deli", type=EntityType.ORGANIZATION)
    assert r._apply_disqualifiers(ExtractedEntity("John", EntityType.PERSON), candidate, 0.9) == 0.0


def test_disqualifier_type_compatible(resolver):
    r, _ = resolver
    from memento.models import Entity
    candidate = Entity(name="John Smith", type=EntityType.PERSON)
    assert r._apply_disqualifiers(ExtractedEntity("John", EntityType.PERSON), candidate, 0.9) == 0.9
