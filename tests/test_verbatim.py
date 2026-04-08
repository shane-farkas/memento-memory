"""Stage 4 tests: Verbatim store with hybrid search."""

from __future__ import annotations

import numpy as np
import pytest

from memento.db import Database
from memento.verbatim_store import VerbatimStore


class FakeEmbedder:
    """Deterministic test embedder using simple hash-based vectors."""

    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> np.ndarray:
        # Create a deterministic embedding from text content
        # Similar texts will produce similar vectors (by shared words)
        words = set(text.lower().split())
        vec = np.zeros(8, dtype=np.float32)
        for word in words:
            h = hash(word) % 8
            vec[h] += 1.0
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def vstore():
    db = Database(":memory:")
    embedder = FakeEmbedder()
    store = VerbatimStore(db, embedder)
    yield store
    db.close()


# ── Basic Store/Retrieve ─────────────────────────────────────────


def test_store_and_retrieve_by_conversation(vstore):
    """Store chunks and retrieve by conversation ID."""
    vstore.store("Hello, how are you?", conversation_id="conv-1", turn_number=1)
    vstore.store("I'm doing great!", conversation_id="conv-1", turn_number=2)
    vstore.store("Different conversation", conversation_id="conv-2", turn_number=1)

    results = vstore.get_by_conversation("conv-1")
    assert len(results) == 2
    assert results[0].turn_number == 1
    assert results[1].turn_number == 2


def test_store_returns_chunk_id(vstore):
    chunk_id = vstore.store("Some text")
    assert chunk_id
    assert len(chunk_id) > 0


# ── FTS5 Keyword Search ─────────────────────────────────────────


def test_fts_search_exact_phrase(vstore):
    """FTS5 should find exact phrases."""
    vstore.store("John Smith works at Acme Corp as Director of Sales")
    vstore.store("Jane Doe is a software engineer at Beta Inc")
    vstore.store("The weather is nice today")

    results = vstore._fts_search("John Smith", top_k=5)
    assert len(results) >= 1
    assert "John Smith" in results[0].text


def test_fts_search_keyword(vstore):
    """FTS5 should match keywords."""
    vstore.store("The project deadline is next Friday")
    vstore.store("We need to review the budget proposal")
    vstore.store("Friday meeting with the team")

    results = vstore._fts_search("Friday", top_k=5)
    assert len(results) >= 1
    # All results should contain "Friday"
    for r in results:
        assert "Friday" in r.text


# ── Vector Search ────────────────────────────────────────────────


def test_vector_search(vstore):
    """Vector search should find semantically similar content."""
    vstore.store("John works at Acme as a sales director")
    vstore.store("The cat sat on the mat")
    vstore.store("John is the director of sales at Acme Corp")

    results = vstore._vector_search("John Acme sales", top_k=3)
    assert len(results) >= 1
    # The John/Acme related results should score higher than the cat


# ── Hybrid Search ────────────────────────────────────────────────


def test_hybrid_search(vstore):
    """Hybrid search combines vector and FTS results."""
    vstore.store("John Smith works at Acme Corp")
    vstore.store("The weather forecast for tomorrow")
    vstore.store("Acme Corp announced quarterly results")
    vstore.store("Something completely unrelated about cooking")

    results = vstore.search("John Smith Acme", top_k=4)
    assert len(results) >= 1
    # Relevant results should be in the returned set
    texts = [r.text for r in results]
    assert any("John" in t or "Acme" in t for t in texts)


def test_hybrid_search_empty_query(vstore):
    """Hybrid search handles edge cases."""
    vstore.store("Some stored text")
    results = vstore.search("", top_k=3)
    # Should not crash, may return results or empty


# ── Multiple Chunks ──────────────────────────────────────────────


def test_store_many_chunks(vstore):
    """Store 10 turns and search them."""
    turns = [
        "I met John Smith today at the conference",
        "He works at Acme Corp as Director of Sales",
        "We discussed the partnership proposal",
        "John mentioned concerns about Q2 performance",
        "He seemed interested in our product",
        "We should follow up next week",
        "The conference was about AI and automation",
        "Other speakers included Jane from Beta Inc",
        "The keynote was about large language models",
        "I also met Sarah from the marketing team",
    ]
    for i, text in enumerate(turns):
        vstore.store(text, conversation_id="conf-day1", turn_number=i + 1)

    # Search for John-related content
    results = vstore.search("John Smith Acme", top_k=5)
    assert len(results) >= 1

    # Search for conference content
    results = vstore.search("conference AI speakers", top_k=5)
    assert len(results) >= 1

    # Get all conversation turns
    all_turns = vstore.get_by_conversation("conf-day1")
    assert len(all_turns) == 10
