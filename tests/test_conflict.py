"""Stage 8 tests: Conflict detection."""

from __future__ import annotations

import pytest

from memento.conflict import ConflictDetector, ConflictType
from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType


@pytest.fixture
def setup():
    db = Database(":memory:")
    graph = GraphStore(db)
    detector = ConflictDetector(graph)
    yield graph, detector
    db.close()


def test_new_fact_no_conflict(setup):
    """First time setting a property → confirmation (no existing value)."""
    graph, detector = setup
    entity = graph.create_entity("John", EntityType.PERSON)

    result = detector.check(entity.id, "location", "Seattle")
    assert result.conflict_type == ConflictType.CONFIRMATION
    assert result.existing_value is None


def test_same_value_confirmation(setup):
    """Same value again → confirmation, confidence boosted."""
    graph, detector = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    graph.set_property(entity.id, "location", "Seattle", confidence=0.8)

    result = detector.check(entity.id, "location", "Seattle")
    assert result.conflict_type == ConflictType.CONFIRMATION

    # Confidence should have been boosted
    prop = graph.get_property(entity.id, "location")
    assert prop.confidence > 0.8


def test_same_value_case_insensitive(setup):
    """Case-insensitive match → confirmation."""
    graph, detector = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    graph.set_property(entity.id, "location", "Seattle")

    result = detector.check(entity.id, "location", "seattle")
    assert result.conflict_type == ConflictType.CONFIRMATION


def test_update_newer_high_authority(setup):
    """Newer timestamp + high authority → update."""
    graph, detector = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    graph.set_property(entity.id, "location", "Seattle", as_of="2025-01-01T00:00:00Z")

    result = detector.check(
        entity.id, "location", "Austin",
        new_as_of="2026-01-01T00:00:00Z",
        new_authority=0.9,
    )
    assert result.conflict_type == ConflictType.UPDATE


def test_historical_older_timestamp(setup):
    """Older timestamp → historical, don't supersede."""
    graph, detector = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    graph.set_property(entity.id, "location", "Austin", as_of="2026-01-01T00:00:00Z")

    result = detector.check(
        entity.id, "location", "Seattle",
        new_as_of="2025-01-01T00:00:00Z",
    )
    assert result.conflict_type == ConflictType.HISTORICAL


def test_contradiction_ambiguous(setup):
    """Ambiguous conflict → creates Conflict record."""
    graph, detector = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    graph.set_property(entity.id, "title", "Director of Sales", as_of="2025-06-01T00:00:00Z")

    result = detector.check(
        entity.id, "title", "VP of Sales",
        new_as_of="2025-07-01T00:00:00Z",
        new_authority=0.5,  # Low authority → ambiguous
    )
    assert result.conflict_type == ConflictType.CONTRADICTION
    assert result.conflict_id is not None

    # Conflict should be recorded
    conflicts = detector.get_unresolved(entity.id)
    assert len(conflicts) == 1
    assert conflicts[0].property_key == "title"


def test_resolve_conflict(setup):
    graph, detector = setup
    entity = graph.create_entity("John", EntityType.PERSON)
    graph.set_property(entity.id, "title", "Director")

    result = detector.check(entity.id, "title", "VP", new_authority=0.5)
    assert result.conflict_type == ConflictType.CONTRADICTION

    detector.resolve(result.conflict_id, "User confirmed VP is correct")

    unresolved = detector.get_unresolved(entity.id)
    assert len(unresolved) == 0
