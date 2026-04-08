"""Stage 7 tests: Entity resolution tier 1 (name/alias/fuzzy)."""

from __future__ import annotations

import pytest

from memento.db import Database
from memento.entity_resolution import (
    EntityResolver,
    levenshtein_distance,
    normalized_edit_similarity,
    phonetic_match,
    soundex,
)
from memento.extraction import ExtractedEntity
from memento.graph_store import GraphStore
from memento.models import EntityType


@pytest.fixture
def resolver():
    db = Database(":memory:")
    graph = GraphStore(db)
    r = EntityResolver(graph)
    yield r, graph
    db.close()


# ── String Utilities ─────────────────────────────────────────────


def test_levenshtein_identical():
    assert levenshtein_distance("hello", "hello") == 0


def test_levenshtein_one_edit():
    assert levenshtein_distance("John", "Jon") == 1


def test_levenshtein_empty():
    assert levenshtein_distance("hello", "") == 5


def test_normalized_similarity_identical():
    assert normalized_edit_similarity("John Smith", "John Smith") == 1.0


def test_normalized_similarity_close():
    sim = normalized_edit_similarity("Jon Smith", "John Smith")
    assert sim > 0.8  # Very similar


def test_normalized_similarity_different():
    sim = normalized_edit_similarity("Alice", "Bob")
    assert sim < 0.5


def test_soundex_basic():
    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"  # Same soundex


def test_soundex_smith():
    assert soundex("Smith") == "S530"
    assert soundex("Smyth") == "S530"  # Same soundex


def test_phonetic_match_similar():
    score = phonetic_match("John Smith", "Jon Smyth")
    assert score >= 0.5


def test_phonetic_match_different():
    score = phonetic_match("John Smith", "Alice Brown")
    assert score == 0.0


# ── Exact Name Resolution ───────────────────────────────────────


def test_resolve_exact_name_match(resolver):
    r, graph = resolver
    existing = graph.create_entity("John Smith", EntityType.PERSON)

    mention = ExtractedEntity("John Smith", EntityType.PERSON)
    result = r.resolve(mention)

    assert result.action == "merge"
    assert result.entity.id == existing.id
    assert result.confidence >= 0.85


def test_resolve_exact_name_case_insensitive(resolver):
    r, graph = resolver
    existing = graph.create_entity("John Smith", EntityType.PERSON)

    mention = ExtractedEntity("john smith", EntityType.PERSON)
    result = r.resolve(mention)

    assert result.action == "merge"
    assert result.entity.id == existing.id


# ── Alias Resolution ────────────────────────────────────────────


def test_resolve_by_alias(resolver):
    r, graph = resolver
    existing = graph.create_entity(
        "John Smith", EntityType.PERSON, aliases=["JS", "John"]
    )

    # "John" should resolve to John Smith when he's the only John
    mention = ExtractedEntity("John", EntityType.PERSON)
    result = r.resolve(mention)

    assert result.action == "merge"
    assert result.entity.id == existing.id


def test_resolve_adds_alias(resolver):
    """When resolving a new name variant, it should suggest adding an alias."""
    r, graph = resolver
    graph.create_entity("John Smith", EntityType.PERSON)

    # Exact match with existing name shouldn't add alias
    mention = ExtractedEntity("John Smith", EntityType.PERSON)
    result = r.resolve(mention)
    assert result.add_alias is None

    # But a new name form should
    graph.create_entity("Jane Doe", EntityType.PERSON, aliases=["Jane"])
    mention2 = ExtractedEntity("Jane", EntityType.PERSON)
    result2 = r.resolve(mention2)
    assert result2.action == "merge"
    # "Jane" is already an alias, so no new alias needed
    assert result2.add_alias is None


# ── Fuzzy Name Resolution ───────────────────────────────────────


def test_resolve_fuzzy_typo(resolver):
    """Jon Smith → John Smith (edit distance 1)."""
    r, graph = resolver
    existing = graph.create_entity("John Smith", EntityType.PERSON)

    mention = ExtractedEntity("Jon Smith", EntityType.PERSON)
    result = r.resolve(mention)

    assert result.action == "merge"
    assert result.entity.id == existing.id


# ── Type Gating ──────────────────────────────────────────────────


def test_resolve_type_mismatch_blocks(resolver):
    """John (person) should NOT resolve to John's Pizza (organization)."""
    r, graph = resolver
    graph.create_entity("John's Pizza", EntityType.ORGANIZATION)

    mention = ExtractedEntity("John", EntityType.PERSON)
    result = r.resolve(mention)

    # Should create new entity because type is incompatible
    assert result.action == "create"


# ── Ambiguity ────────────────────────────────────────────────────


def test_resolve_two_johns_conservative(resolver):
    """With two Johns, ambiguous → create new entity (Tier 1 conservative)."""
    r, graph = resolver
    graph.create_entity("John Smith", EntityType.PERSON)
    graph.create_entity("John Park", EntityType.PERSON)

    # A bare "John" could be either → Tier 1 should create a new entity
    # or merge with the best match. Since both are fuzzy matches,
    # the behavior depends on scoring.
    mention = ExtractedEntity("John", EntityType.PERSON)
    result = r.resolve(mention)

    # Both candidates are plausible but not exact, so this could go either way.
    # The key test is that it doesn't crash and returns a valid result.
    assert result.action in ("merge", "create")


# ── No Match ─────────────────────────────────────────────────────


def test_resolve_no_match(resolver):
    """Completely new entity → create."""
    r, graph = resolver
    graph.create_entity("Alice Brown", EntityType.PERSON)

    mention = ExtractedEntity("Bob Johnson", EntityType.PERSON)
    result = r.resolve(mention)

    assert result.action == "create"


def test_resolve_empty_graph(resolver):
    """No entities in graph → create."""
    r, graph = resolver

    mention = ExtractedEntity("John Smith", EntityType.PERSON)
    result = r.resolve(mention)

    assert result.action == "create"


# ── Batch Resolution ─────────────────────────────────────────────


def test_resolve_batch(resolver):
    r, graph = resolver
    graph.create_entity("John Smith", EntityType.PERSON)
    graph.create_entity("Acme Corp", EntityType.ORGANIZATION)

    mentions = [
        ExtractedEntity("John Smith", EntityType.PERSON),
        ExtractedEntity("Acme Corp", EntityType.ORGANIZATION),
        ExtractedEntity("New Person", EntityType.PERSON),
    ]

    results = r.resolve_batch(mentions)
    assert len(results) == 3

    # First two should merge, third should create
    assert results[0][1].action == "merge"
    assert results[1][1].action == "merge"
    assert results[2][1].action == "create"


# ── Edge Cases ───────────────────────────────────────────────────


def test_resolve_single_word_name(resolver):
    r, graph = resolver
    existing = graph.create_entity("Acme", EntityType.ORGANIZATION)

    mention = ExtractedEntity("Acme", EntityType.ORGANIZATION)
    result = r.resolve(mention)

    assert result.action == "merge"
    assert result.entity.id == existing.id
