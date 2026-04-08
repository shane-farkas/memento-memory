"""Stage 3 tests: Entity merge/split and CLI."""

from __future__ import annotations

import pytest

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType


@pytest.fixture
def store():
    db = Database(":memory:")
    gs = GraphStore(db)
    yield gs
    db.close()


# ── Merge ────────────────────────────────────────────────────────


def test_merge_basic(store):
    """Merge two entities; survivor gets combined aliases."""
    john_a = store.create_entity("John Smith", EntityType.PERSON, aliases=["John"])
    john_b = store.create_entity("JS", EntityType.PERSON, aliases=["J. Smith"])

    result = store.merge_entities(john_a.id, john_b.id)
    assert result["survivor_id"] == john_a.id
    assert result["absorbed_id"] == john_b.id

    survivor = store.get_entity(john_a.id)
    assert survivor is not None
    # Survivor should have aliases from both entities + absorbed name
    assert "JS" in survivor.aliases
    assert "J. Smith" in survivor.aliases
    assert "John" in survivor.aliases

    # Absorbed entity should be archived
    absorbed = store.get_entity(john_b.id)
    assert absorbed is None  # archived entities are hidden


def test_merge_shared_edge(store):
    """Merge two entities that share an edge to a third entity."""
    john_a = store.create_entity("John A", EntityType.PERSON)
    john_b = store.create_entity("John B", EntityType.PERSON)
    acme = store.create_entity("Acme Corp", EntityType.ORGANIZATION)

    store.create_relationship(john_a.id, acme.id, "works_at", confidence=0.8)
    store.create_relationship(john_b.id, acme.id, "works_at", confidence=0.9)

    result = store.merge_entities(john_a.id, john_b.id)

    # Survivor should have one deduplicated edge to Acme
    rels = store.get_relationships(john_a.id, type="works_at")
    assert len(rels) == 1
    # Should keep the higher confidence
    assert rels[0].confidence >= 0.9


def test_merge_properties(store):
    """Merge entities with properties; conflicting props handled."""
    john_a = store.create_entity("John A", EntityType.PERSON)
    john_b = store.create_entity("John B", EntityType.PERSON)

    store.set_property(john_a.id, "title", "Director")
    store.set_property(john_b.id, "location", "Austin")
    store.set_property(john_b.id, "title", "VP")

    result = store.merge_entities(john_a.id, john_b.id)
    assert result["properties_moved"] > 0

    survivor = store.get_entity(john_a.id)
    # Should have both properties; title conflict resolved
    assert "location" in survivor.properties
    assert survivor.properties["location"].value == "Austin"


def test_merge_access_metadata(store):
    """Access counts and timestamps are combined on merge."""
    john_a = store.create_entity("John A", EntityType.PERSON)
    john_b = store.create_entity("John B", EntityType.PERSON)

    # Manually set access counts via direct SQL (the entities start at 0)
    store.db.execute(
        "UPDATE entities SET access_count = 5 WHERE id = ?", (john_a.id,)
    )
    store.db.execute(
        "UPDATE entities SET access_count = 3 WHERE id = ?", (john_b.id,)
    )
    store.db.conn.commit()

    store.merge_entities(john_a.id, john_b.id)
    survivor = store.get_entity(john_a.id)
    assert survivor.access_count == 8


# ── Split (undo merge) ──────────────────────────────────────────


def test_split_basic(store):
    """Merge then split; both entities restored."""
    john = store.create_entity("John Smith", EntityType.PERSON, aliases=["John"])
    js = store.create_entity("JS", EntityType.PERSON, aliases=["J. Smith"])

    acme = store.create_entity("Acme", EntityType.ORGANIZATION)
    store.create_relationship(js.id, acme.id, "works_at")
    store.set_property(js.id, "location", "Austin")

    merge_result = store.merge_entities(john.id, js.id)
    split_result = store.split_entity(merge_result["merge_log_id"])

    assert split_result["survivor_id"] == john.id
    assert split_result["restored_id"] == js.id

    # Both entities should exist now
    restored = store.get_entity(js.id)
    assert restored is not None
    assert restored.name == "JS"

    survivor = store.get_entity(john.id)
    assert survivor is not None

    # Survivor aliases should be restored to original
    assert set(survivor.aliases) == {"John"}


def test_split_nonexistent(store):
    with pytest.raises(ValueError, match="Merge log entry not found"):
        store.split_entity("nonexistent")


# ── CLI ──────────────────────────────────────────────────────────


def test_cli_stats(tmp_path, capsys):
    """CLI stats command works."""
    from memento.cli import main

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    cli_store = GraphStore(db)
    j = cli_store.create_entity("John", EntityType.PERSON)
    a = cli_store.create_entity("Acme", EntityType.ORGANIZATION)
    cli_store.create_relationship(j.id, a.id, "works_at")
    db.close()

    main(["--db", str(db_path), "stats"])
    captured = capsys.readouterr()
    assert "Entities:" in captured.out
    assert "2" in captured.out


def test_cli_entities(tmp_path, capsys):
    from memento.cli import main

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    cli_store = GraphStore(db)
    cli_store.create_entity("John Smith", EntityType.PERSON)
    cli_store.create_entity("Acme Corp", EntityType.ORGANIZATION)
    db.close()

    main(["--db", str(db_path), "entities"])
    captured = capsys.readouterr()
    assert "John Smith" in captured.out
    assert "Acme Corp" in captured.out


def test_cli_entity_detail(tmp_path, capsys):
    from memento.cli import main

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    cli_store = GraphStore(db)
    john = cli_store.create_entity("John Smith", EntityType.PERSON, aliases=["JS"])
    cli_store.set_property(john.id, "title", "Director")
    db.close()

    main(["--db", str(db_path), "entity", john.id])
    captured = capsys.readouterr()
    assert "John Smith" in captured.out
    assert "JS" in captured.out
    assert "title" in captured.out
    assert "Director" in captured.out
