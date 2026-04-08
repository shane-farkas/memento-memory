"""Stage 1 tests: schema creation and model round-trips."""

from __future__ import annotations

import json

from memento.db import Database
from memento.models import (
    Entity,
    EntityType,
    PropertyValue,
    Relationship,
    SourceRef,
    StatementType,
)
from memento.schema import SCHEMA_VERSION, create_tables


def test_schema_creation_idempotent(db):
    """Schema can be created multiple times without error."""
    create_tables(db)
    create_tables(db)
    row = db.fetchone("SELECT version FROM schema_version")
    assert row["version"] == SCHEMA_VERSION


def test_all_tables_exist(db):
    create_tables(db)
    tables = {
        row["name"]
        for row in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    expected = {
        "entities",
        "entity_aliases",
        "source_refs",
        "properties",
        "relationships",
        "relationship_properties",
        "merge_log",
        "conflicts",
        "schema_version",
    }
    assert expected.issubset(tables)


def test_expected_indexes_exist(db):
    create_tables(db)
    indexes = {
        row["name"]
        for row in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
    }
    expected_indexes = {
        "idx_entities_name",
        "idx_entities_type",
        "idx_aliases_entity",
        "idx_aliases_alias",
        "idx_source_refs_conversation",
        "idx_properties_entity_key",
        "idx_rel_source",
        "idx_rel_target",
        "idx_rel_type",
        "idx_merge_log_survivor",
        "idx_conflicts_entity",
        "idx_conflicts_status",
    }
    assert expected_indexes.issubset(indexes)


def test_entity_round_trip(db):
    create_tables(db)
    entity = Entity(name="John Smith", type=EntityType.PERSON)

    db.execute(
        """INSERT INTO entities (id, type, name, created_at, last_seen, access_count, confidence, archived)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entity.id,
            entity.type.value,
            entity.name,
            entity.created_at,
            entity.last_seen,
            entity.access_count,
            entity.confidence,
            int(entity.archived),
        ),
    )
    db.conn.commit()

    row = db.fetchone("SELECT * FROM entities WHERE id = ?", (entity.id,))
    assert row["name"] == "John Smith"
    assert row["type"] == "person"
    assert row["confidence"] == 1.0
    assert row["archived"] == 0


def test_source_ref_round_trip(db):
    create_tables(db)
    ref = SourceRef(
        conversation_id="conv-1",
        turn_number=3,
        statement_type=StatementType.EXPLICIT,
        verbatim="John works at Acme",
        authority=0.9,
    )

    db.execute(
        """INSERT INTO source_refs (id, conversation_id, turn_number, timestamp, statement_type, verbatim, authority)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            ref.id,
            ref.conversation_id,
            ref.turn_number,
            ref.timestamp,
            ref.statement_type.value,
            ref.verbatim,
            ref.authority,
        ),
    )
    db.conn.commit()

    row = db.fetchone("SELECT * FROM source_refs WHERE id = ?", (ref.id,))
    assert row["conversation_id"] == "conv-1"
    assert row["turn_number"] == 3
    assert row["verbatim"] == "John works at Acme"
    assert row["authority"] == 0.9


def test_property_value_round_trip(db):
    create_tables(db)
    # Create parent entity first
    entity = Entity(name="John", type=EntityType.PERSON)
    db.execute(
        """INSERT INTO entities (id, type, name, created_at, last_seen)
           VALUES (?, ?, ?, ?, ?)""",
        (entity.id, entity.type.value, entity.name, entity.created_at, entity.last_seen),
    )

    prop = PropertyValue(
        entity_id=entity.id,
        key="title",
        value="Director of Sales",
        confidence=0.9,
    )

    db.execute(
        """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            prop.id,
            prop.entity_id,
            prop.key,
            json.dumps(prop.value),
            prop.as_of,
            prop.recorded_at,
            prop.confidence,
        ),
    )
    db.conn.commit()

    row = db.fetchone("SELECT * FROM properties WHERE id = ?", (prop.id,))
    assert row["key"] == "title"
    assert json.loads(row["value_json"]) == "Director of Sales"
    assert row["confidence"] == 0.9
    assert row["superseded_by_id"] is None


def test_relationship_round_trip(db):
    create_tables(db)
    # Create two entities
    john = Entity(name="John", type=EntityType.PERSON)
    acme = Entity(name="Acme Corp", type=EntityType.ORGANIZATION)
    for e in [john, acme]:
        db.execute(
            """INSERT INTO entities (id, type, name, created_at, last_seen)
               VALUES (?, ?, ?, ?, ?)""",
            (e.id, e.type.value, e.name, e.created_at, e.last_seen),
        )

    rel = Relationship(
        source_id=john.id,
        target_id=acme.id,
        type="works_at",
        confidence=0.95,
    )

    db.execute(
        """INSERT INTO relationships (id, source_id, target_id, type, valid_from, confidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (rel.id, rel.source_id, rel.target_id, rel.type, rel.valid_from, rel.confidence),
    )
    db.conn.commit()

    row = db.fetchone("SELECT * FROM relationships WHERE id = ?", (rel.id,))
    assert row["source_id"] == john.id
    assert row["target_id"] == acme.id
    assert row["type"] == "works_at"
    assert row["confidence"] == 0.95
    assert row["valid_to"] is None


def test_alias_round_trip(db):
    create_tables(db)
    entity = Entity(name="John Smith", type=EntityType.PERSON)
    db.execute(
        """INSERT INTO entities (id, type, name, created_at, last_seen)
           VALUES (?, ?, ?, ?, ?)""",
        (entity.id, entity.type.value, entity.name, entity.created_at, entity.last_seen),
    )

    from memento.models import _new_id, _now

    alias_id = _new_id()
    db.execute(
        "INSERT INTO entity_aliases (id, entity_id, alias, added_at) VALUES (?, ?, ?, ?)",
        (alias_id, entity.id, "JS", _now()),
    )
    db.conn.commit()

    aliases = db.fetchall(
        "SELECT alias FROM entity_aliases WHERE entity_id = ?", (entity.id,)
    )
    assert len(aliases) == 1
    assert aliases[0]["alias"] == "JS"


def test_entity_type_constraint(db):
    """Invalid entity type should be rejected."""
    create_tables(db)
    import sqlite3

    try:
        db.execute(
            """INSERT INTO entities (id, type, name, created_at, last_seen)
               VALUES ('test', 'invalid_type', 'Test', '2026-01-01', '2026-01-01')""",
        )
        db.conn.commit()
        assert False, "Should have raised an error"
    except sqlite3.IntegrityError:
        pass


def test_foreign_key_constraint(db):
    """Property referencing nonexistent entity should be rejected."""
    create_tables(db)
    import sqlite3

    try:
        db.execute(
            """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at)
               VALUES ('prop1', 'nonexistent', 'key', '"val"', '2026-01-01', '2026-01-01')""",
        )
        db.conn.commit()
        assert False, "Should have raised an error"
    except sqlite3.IntegrityError:
        pass
