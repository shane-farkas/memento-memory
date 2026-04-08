"""Stage 12 tests: Consolidation engine."""

from __future__ import annotations

import pytest

from memento.consolidation import ConsolidationEngine
from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType


@pytest.fixture
def setup():
    db = Database(":memory:")
    graph = GraphStore(db)
    engine = ConsolidationEngine(graph)
    yield graph, engine
    db.close()


# ── Confidence Decay ─────────────────────────────────────────────


def test_decay_recent_fact_unchanged(setup):
    """A just-created fact should barely decay."""
    graph, engine = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    graph.set_property(entity.id, "title", "Director", confidence=0.9)

    engine.decay_confidence()

    prop = graph.get_property(entity.id, "title")
    # Should be very close to original (created just now)
    assert prop.confidence >= 0.85


def test_decay_old_fact(setup):
    """A fact with an old recorded_at should decay."""
    graph, engine = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    prop = graph.set_property(entity.id, "title", "Director", confidence=0.9)

    # Manually backdate the recorded_at to 1 year ago
    graph.db.execute(
        "UPDATE properties SET recorded_at = '2025-04-01T00:00:00+00:00' WHERE id = ?",
        (prop.id,),
    )
    graph.db.conn.commit()

    decayed = engine.decay_confidence()
    assert decayed >= 1

    updated_prop = graph.get_property(entity.id, "title")
    assert updated_prop.confidence < 0.9


# ── Redundancy Merging ───────────────────────────────────────────


def test_merge_duplicate_facts(setup):
    """Two identical property values → merged into one."""
    graph, engine = setup
    entity = graph.create_entity("John", EntityType.PERSON)

    # Insert two non-superseded values for the same key with same value
    # (simulating ingestion from two different conversations)
    from memento.models import _new_id, _now
    import json

    id1 = _new_id()
    id2 = _new_id()
    now = _now()
    graph.db.execute(
        """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (id1, entity.id, "location", json.dumps("Seattle"), now, "2026-01-01T00:00:00+00:00", 0.8),
    )
    graph.db.execute(
        """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (id2, entity.id, "location", json.dumps("Seattle"), now, "2026-02-01T00:00:00+00:00", 0.8),
    )
    graph.db.conn.commit()

    merged = engine.merge_redundant()
    assert merged >= 1

    # Only one non-superseded value should remain
    remaining = graph.db.fetchall(
        "SELECT * FROM properties WHERE entity_id = ? AND key = ? AND superseded_by_id IS NULL",
        (entity.id, "location"),
    )
    assert len(remaining) == 1
    # Confidence should be boosted
    assert remaining[0]["confidence"] > 0.8


def test_no_merge_different_values(setup):
    """Different values for same key → no merge."""
    graph, engine = setup
    entity = graph.create_entity("John", EntityType.PERSON)

    from memento.models import _new_id, _now
    import json

    now = _now()
    graph.db.execute(
        """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (_new_id(), entity.id, "location", json.dumps("Seattle"), now, now, 0.8),
    )
    graph.db.execute(
        """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (_new_id(), entity.id, "location", json.dumps("Austin"), now, now, 0.8),
    )
    graph.db.conn.commit()

    merged = engine.merge_redundant()
    assert merged == 0


# ── Orphan Pruning ───────────────────────────────────────────────


def test_prune_orphan(setup):
    """Entity with no edges, low confidence, old last_seen → archived."""
    graph, engine = setup
    entity = graph.create_entity("Orphan", EntityType.CONCEPT)

    # Set low confidence and old last_seen
    graph.db.execute(
        "UPDATE entities SET confidence = 0.1, last_seen = '2025-01-01T00:00:00+00:00' WHERE id = ?",
        (entity.id,),
    )
    graph.db.conn.commit()

    pruned = engine.prune_orphans()
    assert pruned >= 1

    # Should be archived
    fetched = graph.get_entity(entity.id)
    assert fetched is None


def test_no_prune_connected_entity(setup):
    """Entity with edges should not be pruned even if low confidence."""
    graph, engine = setup
    entity = graph.create_entity("Connected", EntityType.PERSON)
    other = graph.create_entity("Other", EntityType.PERSON)
    graph.create_relationship(entity.id, other.id, "knows")

    graph.db.execute(
        "UPDATE entities SET confidence = 0.1, last_seen = '2025-01-01T00:00:00+00:00' WHERE id = ?",
        (entity.id,),
    )
    graph.db.conn.commit()

    pruned = engine.prune_orphans()
    assert pruned == 0

    fetched = graph.get_entity(entity.id)
    assert fetched is not None


# ── Contradiction Auto-Resolution ────────────────────────────────


def test_auto_resolve_decayed_contradiction(setup):
    graph, engine = setup
    entity = graph.create_entity("John", EntityType.PERSON)

    # Create two property values and a conflict
    from memento.models import _new_id, _now
    import json

    id_a = _new_id()
    id_b = _new_id()
    now = _now()
    graph.db.execute(
        """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (id_a, entity.id, "title", json.dumps("Director"), now, now, 0.1),  # Decayed
    )
    graph.db.execute(
        """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (id_b, entity.id, "title", json.dumps("VP"), now, now, 0.9),  # Still strong
    )
    graph.db.execute(
        """INSERT INTO conflicts (id, entity_id, property_key, value_a_id, value_b_id, status, created_at)
           VALUES (?, ?, ?, ?, ?, 'unresolved', ?)""",
        (_new_id(), entity.id, "title", id_a, id_b, now),
    )
    graph.db.conn.commit()

    resolved = engine.auto_resolve_contradictions()
    assert resolved >= 1

    # No more unresolved conflicts
    unresolved = graph.db.fetchall(
        "SELECT * FROM conflicts WHERE entity_id = ? AND status = 'unresolved'",
        (entity.id,),
    )
    assert len(unresolved) == 0


# ── Centrality Index ─────────────────────────────────────────────


def test_materialize_centrality(setup):
    graph, engine = setup
    john = graph.create_entity("John", EntityType.PERSON)
    acme = graph.create_entity("Acme", EntityType.ORGANIZATION)
    graph.create_relationship(john.id, acme.id, "works_at")

    count = engine.materialize_centrality()
    assert count >= 2

    # John should have degree > 0
    row = graph.db.fetchone(
        "SELECT degree FROM centrality_index WHERE entity_id = ?", (john.id,)
    )
    assert row is not None
    assert row["degree"] > 0


# ── Full Consolidation Pass ──────────────────────────────────────


def test_full_consolidation(setup):
    graph, engine = setup
    graph.create_entity("John", EntityType.PERSON)
    graph.create_entity("Acme", EntityType.ORGANIZATION)

    result = engine.run_full()
    assert result.centrality_entries >= 2


def test_quick_consolidation(setup):
    graph, engine = setup
    graph.create_entity("John", EntityType.PERSON)

    result = engine.run_quick()
    assert isinstance(result.facts_decayed, int)
    assert isinstance(result.redundancies_merged, int)
