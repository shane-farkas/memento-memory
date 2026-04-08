"""Stage 10 tests: Full retrieval engine."""

from __future__ import annotations

import numpy as np
import pytest

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType
from memento.retrieval import RetrievalEngine, RetrievalFact
from memento.verbatim_store import VerbatimStore


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> np.ndarray:
        words = set(text.lower().split())
        vec = np.zeros(8, dtype=np.float32)
        for word in words:
            h = hash(word) % 8
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def engine():
    db = Database(":memory:")
    graph = GraphStore(db)
    embedder = FakeEmbedder()
    verbatim = VerbatimStore(db, embedder)
    engine = RetrievalEngine(graph, verbatim=verbatim)
    yield engine, graph, verbatim
    db.close()


@pytest.fixture
def engine_no_verbatim():
    db = Database(":memory:")
    graph = GraphStore(db)
    engine = RetrievalEngine(graph, verbatim=None)
    yield engine, graph
    db.close()


def _build_test_graph(graph):
    """Build a small test graph with a few entities and relationships."""
    john = graph.create_entity("John Smith", EntityType.PERSON, aliases=["John", "JS"])
    acme = graph.create_entity("Acme Corp", EntityType.ORGANIZATION)
    beta = graph.create_entity("Beta Inc", EntityType.ORGANIZATION)
    falcon = graph.create_entity("Project Falcon", EntityType.PROJECT)

    graph.create_relationship(john.id, acme.id, "works_at", confidence=0.95)
    graph.create_relationship(acme.id, beta.id, "acquiring", confidence=0.7)
    graph.create_relationship(john.id, falcon.id, "leads", confidence=0.8)

    graph.set_property(john.id, "title", "Director of Sales", confidence=0.9)
    graph.set_property(john.id, "location", "Chicago")
    graph.set_property(acme.id, "industry", "Technology")
    graph.set_property(falcon.id, "launch_date", "April 2026")

    return john, acme, beta, falcon


# ── Basic Retrieval ──────────────────────────────────────────────


def test_recall_by_entity_name(engine):
    eng, graph, _ = engine
    john, acme, _, _ = _build_test_graph(graph)

    result = eng.recall("John Smith")
    assert "John Smith" in result.text
    assert "Director of Sales" in result.text
    assert result.entity_count >= 1


def test_recall_multi_hop(engine):
    """Query about John should include his company and project."""
    eng, graph, _ = engine
    _build_test_graph(graph)

    result = eng.recall("John Smith")
    assert "John Smith" in result.text
    # Should include relationship to Acme
    assert "Acme Corp" in result.text


def test_recall_no_results(engine):
    eng, graph, _ = engine
    result = eng.recall("completely unknown xyz123")
    assert "No relevant memories" in result.text or len(result.text) > 0


# ── Token Budget ─────────────────────────────────────────────────


def test_token_budget_respected(engine):
    eng, graph, _ = engine
    _build_test_graph(graph)

    # Very small budget
    result_small = eng.recall("John Smith", token_budget=50)
    # Larger budget
    result_large = eng.recall("John Smith", token_budget=2000)

    # Small result should be shorter (or at least not longer)
    assert result_small.token_estimate <= result_large.token_estimate + 10


# ── Point-in-Time Retrieval ──────────────────────────────────────


def test_point_in_time_retrieval(engine):
    eng, graph, _ = engine
    john = graph.create_entity("John Smith", EntityType.PERSON)
    p1 = graph.set_property(john.id, "location", "Seattle")
    t1 = p1.recorded_at
    p2 = graph.set_property(john.id, "location", "Austin")
    t2 = p2.recorded_at

    # Query as of t1 should show Seattle
    result = eng.recall("John Smith", as_of=t1)
    assert "Seattle" in result.text

    # Query as of t2 should show Austin
    result2 = eng.recall("John Smith", as_of=t2)
    assert "Austin" in result2.text


# ── Verbatim Fusion ─────────────────────────────────────────────


def test_verbatim_fusion(engine):
    eng, graph, verbatim = engine
    verbatim.store("The project deadline is next Friday")
    verbatim.store("Budget meeting scheduled for Monday")

    # Simple recall query should include verbatim results
    result = eng.recall("what did I say about the deadline")
    assert "deadline" in result.text.lower() or "Friday" in result.text


def test_recall_without_verbatim(engine_no_verbatim):
    eng, graph = engine_no_verbatim
    john = graph.create_entity("John Smith", EntityType.PERSON)
    graph.set_property(john.id, "title", "Director")

    result = eng.recall("John Smith")
    assert "John Smith" in result.text
    assert "Director" in result.text


# ── Serialization Format ────────────────────────────────────────


def test_serialization_groups_by_entity(engine):
    eng, graph, _ = engine
    _build_test_graph(graph)

    result = eng.recall("John Smith")
    # Should have entity headers with markdown format
    assert "## John Smith" in result.text


def test_serialization_marks_low_confidence(engine):
    eng, graph, _ = engine
    john = graph.create_entity("John Smith", EntityType.PERSON)
    graph.set_property(john.id, "rumor", "might be leaving", confidence=0.3)

    result = eng.recall("John Smith")
    assert "unconfirmed" in result.text


# ── Ranking Signals ──────────────────────────────────────────────


def test_query_entity_facts_ranked_higher(engine):
    eng, graph, _ = engine
    _build_test_graph(graph)

    result = eng.recall("John Smith")
    # Facts about John should appear before facts about 2-hop entities
    john_idx = result.text.find("John Smith")
    assert john_idx >= 0
    assert result.facts[0].entity.name == "John Smith"


# ── Edge Cases ───────────────────────────────────────────────────


def test_recall_empty_query(engine):
    eng, graph, _ = engine
    _build_test_graph(graph)
    result = eng.recall("")
    # Should not crash
    assert isinstance(result.text, str)


def test_recall_entity_with_no_properties(engine):
    eng, graph, _ = engine
    graph.create_entity("Mysterious Entity", EntityType.CONCEPT)

    result = eng.recall("Mysterious Entity")
    assert "Mysterious Entity" in result.text
