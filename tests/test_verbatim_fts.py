"""Tests for the FTS5 query builder used by verbatim keyword search.

``_build_fts_query`` only reads the class-level stopword set, so a bare
instance (no database) is enough to exercise it directly.
"""

from __future__ import annotations

from memento.verbatim_store import VerbatimStore


def _build(query: str) -> str:
    inst = object.__new__(VerbatimStore)
    return VerbatimStore._build_fts_query(inst, query)


def test_drops_stopwords_keeps_content():
    assert _build("what is the weather") == '"weather"'


def test_ors_content_terms():
    assert _build("favorite italian restaurant") == (
        '"favorite" OR "italian" OR "restaurant"'
    )


def test_dedupes_repeated_terms():
    assert _build("travel travel plans") == '"travel" OR "plans"'


def test_empty_for_no_content_tokens():
    assert _build("hi") == ""
    assert _build("!!!") == ""


def test_punctuation_and_operators_are_neutralized():
    # FTS5 operators / quotes / punctuation must not leak through — only
    # [A-Za-z0-9] tokens survive, each wrapped as a literal quoted term.
    assert _build('rust OR "drop table"; c++') == '"rust" OR "drop" OR "table"'


def test_all_stopword_query_falls_back_to_long_tokens():
    # All stopwords, but "what"/"the" are >2 chars, so the fallback keeps them
    # rather than returning an empty (never-matching) query.
    out = _build("what is the")
    assert out and out.startswith('"')
