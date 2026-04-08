"""Stage 8 tests: Session scratchpad."""

from __future__ import annotations

import pytest

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import EntityType
from memento.scratchpad import SessionScratchpad


@pytest.fixture
def scratchpad():
    db = Database(":memory:")
    graph = GraphStore(db)
    sp = SessionScratchpad(graph=graph)
    yield sp, graph
    db.close()


@pytest.fixture
def scratchpad_no_graph():
    return SessionScratchpad(graph=None)


# ── Name Extraction ──────────────────────────────────────────────


def test_extract_person_name(scratchpad):
    sp, _ = scratchpad
    mentions = sp.on_turn("I met John Smith today at the conference", turn_number=1)
    names = [m.text for m in mentions if m.type_hint == EntityType.PERSON]
    assert "John Smith" in names


def test_extract_org_name(scratchpad):
    sp, _ = scratchpad
    mentions = sp.on_turn("She works at Acme Corp downtown", turn_number=1)
    names = [m.text for m in mentions if m.type_hint == EntityType.ORGANIZATION]
    assert "Acme Corp" in names


def test_no_entities_in_text(scratchpad):
    sp, _ = scratchpad
    mentions = sp.on_turn("the weather is nice today", turn_number=1)
    # Should only have pronouns or nothing
    named = [m for m in mentions if m.type_hint is not None and len(m.text) > 3]
    assert len(named) == 0


# ── Pronoun Coreference ─────────────────────────────────────────


def test_pronoun_coreference(scratchpad):
    sp, _ = scratchpad

    # Turn 1: introduce John Smith
    sp.on_turn("I met John Smith today", turn_number=1)

    # Turn 3: "He" should link to John Smith
    mentions = sp.on_turn("He mentioned a new project", turn_number=3)
    pronoun_mentions = [m for m in mentions if m.text.lower() == "he"]

    assert len(pronoun_mentions) >= 1
    # The pronoun should share the tentative_id with John Smith
    john_id = None
    for m in sp.mentions:
        if m.text == "John Smith":
            john_id = m.tentative_id
            break
    assert john_id is not None
    assert pronoun_mentions[0].tentative_id == john_id


def test_coreference_chains(scratchpad):
    sp, _ = scratchpad

    sp.on_turn("I met John Smith today", turn_number=1)
    sp.on_turn("He mentioned a new project", turn_number=3)
    sp.on_turn("John Smith is excited about it", turn_number=7)

    chains = sp.get_coreference_chains()
    # Should have at least one chain with John Smith + "He" + John Smith again
    assert len(chains) >= 1

    # Find the chain for John Smith
    john_chain = None
    for tid, chain in chains.items():
        texts = [m.text for m in chain]
        if "John Smith" in texts:
            john_chain = chain
            break

    assert john_chain is not None
    assert len(john_chain) >= 2  # At least the original mention + one more


# ── Graph-Aware Resolution ───────────────────────────────────────


def test_resolve_against_graph(scratchpad):
    sp, graph = scratchpad
    # Pre-populate graph
    existing = graph.create_entity("John Smith", EntityType.PERSON)

    mentions = sp.on_turn("I talked to John Smith today", turn_number=1)
    john_mentions = [m for m in mentions if m.text == "John Smith"]

    assert len(john_mentions) >= 1
    assert john_mentions[0].resolved_to == existing.id


def test_same_name_reuses_tentative_id(scratchpad):
    sp, _ = scratchpad

    sp.on_turn("John Smith called me", turn_number=1)
    sp.on_turn("John Smith sent an email too", turn_number=2)

    # Both mentions of John Smith should have the same tentative_id
    john_ids = set()
    for m in sp.mentions:
        if m.text == "John Smith":
            john_ids.add(m.tentative_id)
    assert len(john_ids) == 1


# ── Unique Entities ──────────────────────────────────────────────


def test_get_unique_entities(scratchpad):
    sp, _ = scratchpad

    sp.on_turn("I met John Smith at Acme Corp", turn_number=1)
    sp.on_turn("He works with Jane Doe there", turn_number=2)

    unique = sp.get_unique_entities()
    names = [name for _, name, _ in unique]
    assert "John Smith" in names
    assert "Acme Corp" in names
    assert "Jane Doe" in names


# ── No Graph Mode ────────────────────────────────────────────────


def test_scratchpad_without_graph(scratchpad_no_graph):
    sp = scratchpad_no_graph
    mentions = sp.on_turn("I met John Smith today", turn_number=1)
    assert len(mentions) >= 1
    # Should still extract names, just can't resolve against graph
    john_mentions = [m for m in mentions if m.text == "John Smith"]
    assert john_mentions[0].resolved_to is None
