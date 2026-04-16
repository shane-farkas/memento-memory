"""Stage 2 tests: Graph CRUD operations and bitemporal queries."""

from __future__ import annotations

import time

import pytest

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType, SourceRef


@pytest.fixture
def store():
    db = Database(":memory:")
    gs = GraphStore(db)
    yield gs
    db.close()


# ── Entity CRUD ──────────────────────────────────────────────────


def test_create_entity(store):
    entity = store.create_entity("John Smith", EntityType.PERSON)
    assert entity.name == "John Smith"
    assert entity.type == EntityType.PERSON
    assert entity.confidence == 1.0
    assert entity.id


def test_create_entity_with_aliases(store):
    entity = store.create_entity(
        "John Smith", EntityType.PERSON, aliases=["John", "JS"]
    )
    assert set(entity.aliases) == {"John", "JS"}


def test_get_entity(store):
    created = store.create_entity("John Smith", EntityType.PERSON, aliases=["JS"])
    fetched = store.get_entity(created.id)
    assert fetched is not None
    assert fetched.name == "John Smith"
    assert "JS" in fetched.aliases


def test_get_entity_not_found(store):
    assert store.get_entity("nonexistent") is None


def test_search_entities_exact(store):
    store.create_entity("John Smith", EntityType.PERSON)
    store.create_entity("Jane Doe", EntityType.PERSON)
    results = store.search_entities(name="John Smith")
    assert len(results) == 1
    assert results[0].name == "John Smith"


def test_search_entities_by_alias(store):
    store.create_entity("John Smith", EntityType.PERSON, aliases=["JS"])
    results = store.search_entities(name="JS")
    assert len(results) == 1
    assert results[0].name == "John Smith"


def test_search_entities_fuzzy(store):
    store.create_entity("John Smith", EntityType.PERSON)
    results = store.search_entities(name="John", fuzzy=True)
    assert len(results) == 1


def test_search_entities_by_type(store):
    store.create_entity("John Smith", EntityType.PERSON)
    store.create_entity("Acme Corp", EntityType.ORGANIZATION)
    results = store.search_entities(type=EntityType.PERSON)
    assert len(results) == 1
    assert results[0].name == "John Smith"


# ── Property CRUD with Supersession ─────────────────────────────


def test_set_and_get_property(store):
    entity = store.create_entity("John Smith", EntityType.PERSON)
    store.set_property(entity.id, "title", "Director of Sales")
    prop = store.get_property(entity.id, "title")
    assert prop is not None
    assert prop.value == "Director of Sales"
    assert prop.superseded_by_id is None


def test_property_supersession_chain(store):
    """Update a property twice → supersession chain has 3 nodes."""
    entity = store.create_entity("John Smith", EntityType.PERSON)

    p1 = store.set_property(entity.id, "title", "Manager", as_of="2025-01-01T00:00:00Z")
    p2 = store.set_property(entity.id, "title", "Director", as_of="2025-06-01T00:00:00Z")
    p3 = store.set_property(entity.id, "title", "VP", as_of="2026-01-01T00:00:00Z")

    history = store.get_property_history(entity.id, "title")
    assert len(history) == 3

    # First two should be superseded, last should be current
    assert history[0].superseded_by_id == history[1].id
    assert history[1].superseded_by_id == history[2].id
    assert history[2].superseded_by_id is None

    # Current value should be VP
    current = store.get_property(entity.id, "title")
    assert current.value == "VP"


def test_property_with_source_ref(store):
    entity = store.create_entity("John Smith", EntityType.PERSON)
    ref = SourceRef(
        conversation_id="conv-1",
        turn_number=3,
        verbatim="John is the Director of Sales",
    )
    store.set_property(entity.id, "title", "Director", source_ref=ref)
    prop = store.get_property(entity.id, "title")
    assert prop.source_ref_id == ref.id


# ── Bitemporal Queries ───────────────────────────────────────────


def test_point_in_time_snapshot(store):
    """Property A at t1, update to B at t2. Query as_of t1 returns A, as_of t2 returns B."""
    entity = store.create_entity("John Smith", EntityType.PERSON)

    store.set_property(entity.id, "location", "Seattle")
    # Small sleep to ensure different recorded_at timestamps
    t1 = store.get_property(entity.id, "location").recorded_at

    store.set_property(entity.id, "location", "Austin")
    t2 = store.get_property(entity.id, "location").recorded_at

    # Snapshot at t1 should show Seattle
    snap1 = store.point_in_time_snapshot(entity.id, t1)
    assert snap1 is not None
    assert "location" in snap1.properties
    assert snap1.properties["location"].value == "Seattle"

    # Snapshot at t2 should show Austin
    snap2 = store.point_in_time_snapshot(entity.id, t2)
    assert snap2.properties["location"].value == "Austin"


def test_point_in_time_before_any_facts(store):
    """Snapshot before any facts were recorded returns entity with no properties."""
    entity = store.create_entity("John Smith", EntityType.PERSON)
    store.set_property(entity.id, "location", "Seattle")

    snap = store.point_in_time_snapshot(entity.id, "2020-01-01T00:00:00Z")
    assert snap is not None
    assert len(snap.properties) == 0


# ── Relationship CRUD ────────────────────────────────────────────


def test_create_relationship(store):
    john = store.create_entity("John Smith", EntityType.PERSON)
    acme = store.create_entity("Acme Corp", EntityType.ORGANIZATION)
    rel = store.create_relationship(john.id, acme.id, "works_at")
    assert rel.source_id == john.id
    assert rel.target_id == acme.id
    assert rel.type == "works_at"


def test_get_relationships(store):
    john = store.create_entity("John Smith", EntityType.PERSON)
    acme = store.create_entity("Acme Corp", EntityType.ORGANIZATION)
    beta = store.create_entity("Beta Inc", EntityType.ORGANIZATION)

    store.create_relationship(john.id, acme.id, "works_at")
    store.create_relationship(john.id, beta.id, "invested_in")

    # All relationships
    all_rels = store.get_relationships(john.id)
    assert len(all_rels) == 2

    # Outgoing only
    outgoing = store.get_relationships(john.id, direction="outgoing")
    assert len(outgoing) == 2

    # Filter by type
    works = store.get_relationships(john.id, type="works_at")
    assert len(works) == 1
    assert works[0].type == "works_at"


def test_find_relationship(store):
    john = store.create_entity("John Smith", EntityType.PERSON)
    acme = store.create_entity("Acme Corp", EntityType.ORGANIZATION)
    store.create_relationship(john.id, acme.id, "works_at")

    rel = store.find_relationship(john.id, acme.id, "works_at")
    assert rel is not None
    assert rel.type == "works_at"

    # Non-existent
    assert store.find_relationship(john.id, acme.id, "invested_in") is None


# ── Graph Traversal ──────────────────────────────────────────────


def test_get_neighbors_1_hop(store):
    a = store.create_entity("A", EntityType.PERSON)
    b = store.create_entity("B", EntityType.PERSON)
    c = store.create_entity("C", EntityType.PERSON)
    store.create_relationship(a.id, b.id, "knows")
    store.create_relationship(b.id, c.id, "knows")

    neighbors = store.get_neighbors(a.id, max_hops=1)
    names = {n.name for n in neighbors}
    assert names == {"B"}


def test_get_neighbors_2_hops(store):
    a = store.create_entity("A", EntityType.PERSON)
    b = store.create_entity("B", EntityType.PERSON)
    c = store.create_entity("C", EntityType.PERSON)
    d = store.create_entity("D", EntityType.PERSON)
    store.create_relationship(a.id, b.id, "knows")
    store.create_relationship(b.id, c.id, "knows")
    store.create_relationship(c.id, d.id, "knows")

    neighbors = store.get_neighbors(a.id, max_hops=2)
    names = {n.name for n in neighbors}
    assert names == {"B", "C"}


def test_get_neighbors_3_hops(store):
    a = store.create_entity("A", EntityType.PERSON)
    b = store.create_entity("B", EntityType.PERSON)
    c = store.create_entity("C", EntityType.PERSON)
    d = store.create_entity("D", EntityType.PERSON)
    store.create_relationship(a.id, b.id, "knows")
    store.create_relationship(b.id, c.id, "knows")
    store.create_relationship(c.id, d.id, "knows")

    neighbors = store.get_neighbors(a.id, max_hops=3)
    names = {n.name for n in neighbors}
    assert names == {"B", "C", "D"}


# ── Transaction Atomicity ────────────────────────────────────────


def test_transaction_atomicity(store):
    """Force a failure mid-update, verify no partial writes."""
    entity = store.create_entity("John", EntityType.PERSON)
    store.set_property(entity.id, "location", "Seattle")

    try:
        with store.db.transaction() as cur:
            cur.execute(
                """UPDATE properties SET value_json = ? WHERE entity_id = ? AND key = ? AND superseded_by_id IS NULL""",
                ('"Austin"', entity.id, "location"),
            )
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    # Should still be Seattle
    prop = store.get_property(entity.id, "location")
    assert prop.value == "Seattle"


# ── Stats ────────────────────────────────────────────────────────


def test_stats(store):
    john = store.create_entity("John", EntityType.PERSON)
    acme = store.create_entity("Acme", EntityType.ORGANIZATION)
    store.create_relationship(john.id, acme.id, "works_at")
    store.set_property(john.id, "title", "Director")

    s = store.stats()
    assert s["node_count"] == 2
    assert s["edge_count"] == 1
    assert s["active_property_count"] == 1
    assert s["avg_confidence"] == 1.0


# ── Tier counters ────────────────────────────────────────────────


def test_new_entity_starts_with_zero_counters(store):
    e = store.create_entity("Alice", EntityType.PERSON)
    row = store.db.fetchone(
        "SELECT mention_count, source_count, tier FROM entities WHERE id = ?",
        (e.id,),
    )
    assert row["mention_count"] == 0
    assert row["source_count"] == 0
    assert row["tier"] == 3


def test_record_mention_bumps_counters(store):
    e = store.create_entity("Alice", EntityType.PERSON)
    store.record_mention(e.id, "conv-1")
    row = store.db.fetchone(
        "SELECT mention_count, source_count FROM entities WHERE id = ?",
        (e.id,),
    )
    assert row["mention_count"] == 1
    assert row["source_count"] == 1


def test_record_mention_dedupes_same_conversation(store):
    e = store.create_entity("Alice", EntityType.PERSON)
    store.record_mention(e.id, "conv-1")
    store.record_mention(e.id, "conv-1")
    store.record_mention(e.id, "conv-1")
    row = store.db.fetchone(
        "SELECT mention_count, source_count FROM entities WHERE id = ?",
        (e.id,),
    )
    # mention_count rises every call; source_count is distinct conversations only
    assert row["mention_count"] == 3
    assert row["source_count"] == 1


def test_record_mention_counts_distinct_sources(store):
    e = store.create_entity("Alice", EntityType.PERSON)
    store.record_mention(e.id, "conv-1")
    store.record_mention(e.id, "conv-2")
    store.record_mention(e.id, "conv-3")
    row = store.db.fetchone(
        "SELECT mention_count, source_count FROM entities WHERE id = ?",
        (e.id,),
    )
    assert row["mention_count"] == 3
    assert row["source_count"] == 3


def test_record_mention_ignores_empty_conversation(store):
    e = store.create_entity("Alice", EntityType.PERSON)
    store.record_mention(e.id, "")
    row = store.db.fetchone(
        "SELECT mention_count, source_count FROM entities WHERE id = ?",
        (e.id,),
    )
    assert row["mention_count"] == 0
    assert row["source_count"] == 0
