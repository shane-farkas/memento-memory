"""Stage 13 tests: Privacy/audit layer."""

from __future__ import annotations

import json

import pytest

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType, SourceRef
from memento.privacy import PrivacyLayer


@pytest.fixture
def setup():
    db = Database(":memory:")
    graph = GraphStore(db)
    privacy = PrivacyLayer(graph)
    yield graph, privacy
    db.close()


def _populate_graph(graph):
    ref = SourceRef(conversation_id="conv-1", turn_number=1, verbatim="John works at Acme")
    john = graph.create_entity("John Smith", EntityType.PERSON, aliases=["JS"], source_ref=ref)
    acme = graph.create_entity("Acme Corp", EntityType.ORGANIZATION, source_ref=ref)
    graph.create_relationship(john.id, acme.id, "works_at", source_ref=ref)
    graph.set_property(john.id, "title", "Director", source_ref=ref)
    graph.set_property(john.id, "location", "Chicago", source_ref=ref)
    return john, acme, ref


# ── Access Logging ───────────────────────────────────────────────


def test_access_logging(setup):
    graph, privacy = setup
    john, _, _ = _populate_graph(graph)

    privacy.log_access(john.id, query="who is John", caller="test")
    privacy.log_access(john.id, query="John's title", caller="test")

    rows = graph.db.fetchall(
        "SELECT * FROM access_log WHERE entity_id = ?", (john.id,)
    )
    assert len(rows) == 2


# ── Data Export ──────────────────────────────────────────────────


def test_export_entity_data(setup):
    graph, privacy = setup
    john, _, _ = _populate_graph(graph)
    privacy.log_access(john.id, query="test")

    export = privacy.export_entity_data(john.id)
    assert export is not None
    assert export.name == "John Smith"
    assert export.type == "person"
    assert "JS" in export.aliases
    assert len(export.properties) >= 2  # title + location
    assert len(export.relationships) >= 1  # works_at
    assert len(export.source_refs) >= 1
    assert len(export.access_log) >= 1

    # Should be serializable to JSON
    json_str = export.to_json()
    parsed = json.loads(json_str)
    assert parsed["name"] == "John Smith"


def test_export_nonexistent(setup):
    _, privacy = setup
    export = privacy.export_entity_data("nonexistent")
    assert export is None


# ── Belief Audit ─────────────────────────────────────────────────


def test_audit_belief(setup):
    graph, privacy = setup
    john, _, ref = _populate_graph(graph)

    # Update the title to create a history
    ref2 = SourceRef(conversation_id="conv-2", turn_number=5, verbatim="John is now VP")
    graph.set_property(john.id, "title", "VP", source_ref=ref2)

    chain = privacy.audit_belief(john.id, "title")
    assert chain.entity_id == john.id
    assert chain.property_key == "title"
    assert len(chain.chain) >= 2

    # Most recent first
    assert chain.chain[0]["value"] == "VP"
    assert chain.chain[1]["value"] == "Director"

    # Source provenance should be traced
    assert chain.chain[0]["source"] is not None
    assert chain.chain[0]["source"]["verbatim"] == "John is now VP"


def test_audit_belief_no_history(setup):
    graph, privacy = setup
    john, _, _ = _populate_graph(graph)

    chain = privacy.audit_belief(john.id, "nonexistent_property")
    assert len(chain.chain) == 0


# ── Hard Delete ──────────────────────────────────────────────────


def test_hard_delete(setup):
    graph, privacy = setup
    john, acme, _ = _populate_graph(graph)
    privacy.log_access(john.id, query="test")

    receipt = privacy.delete_entity_cascade(john.id)
    assert receipt is not None
    assert receipt.entity_name == "John Smith"
    assert receipt.content_hash  # Should be a hex string
    assert len(receipt.content_hash) == 64  # SHA-256 hex
    assert receipt.items_deleted["properties"] >= 2
    assert receipt.items_deleted["relationships"] >= 1
    assert receipt.items_deleted["aliases"] >= 1

    # Entity should be completely gone
    assert graph.db.fetchone("SELECT id FROM entities WHERE id = ?", (john.id,)) is None
    assert graph.db.fetchall("SELECT * FROM properties WHERE entity_id = ?", (john.id,)) == []
    assert graph.db.fetchall("SELECT * FROM entity_aliases WHERE entity_id = ?", (john.id,)) == []

    # Access log rows are INTENTIONALLY preserved as an audit trail.
    # See comment in privacy.delete_entity_cascade — wiping them would
    # destroy the record the function is supposed to produce.
    assert graph.db.fetchall("SELECT * FROM access_log WHERE entity_id = ?", (john.id,)) != []

    # A deletion_audit_log row should exist for this entity.
    audit_row = graph.db.fetchone(
        "SELECT * FROM deletion_audit_log WHERE entity_id = ?", (john.id,)
    )
    assert audit_row is not None
    assert audit_row["entity_name"] == "John Smith"
    assert audit_row["content_hash"] == receipt.content_hash

    # Acme should still exist
    assert graph.get_entity(acme.id) is not None


def test_hard_delete_nonexistent(setup):
    _, privacy = setup
    receipt = privacy.delete_entity_cascade("nonexistent")
    assert receipt is None


def test_hard_delete_preserves_shared_source_refs(setup):
    """Source refs used by other entities should NOT be deleted."""
    graph, privacy = setup
    shared_ref = SourceRef(conversation_id="conv-shared", verbatim="Shared context")
    john = graph.create_entity("John", EntityType.PERSON, source_ref=shared_ref)
    acme = graph.create_entity("Acme", EntityType.ORGANIZATION, source_ref=shared_ref)
    graph.set_property(john.id, "title", "Director", source_ref=shared_ref)
    graph.set_property(acme.id, "industry", "Tech", source_ref=shared_ref)

    privacy.delete_entity_cascade(john.id)

    # Shared source ref should still exist (Acme still references it)
    ref_row = graph.db.fetchone(
        "SELECT id FROM source_refs WHERE id = ?", (shared_ref.id,)
    )
    assert ref_row is not None
