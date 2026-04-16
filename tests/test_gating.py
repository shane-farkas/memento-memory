"""Tests for the ingest gate.

Covers:
- PreFilterGate unit behavior (length + acknowledgment patterns)
- End-to-end gate plumbing through MemoryStore.ingest
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from memento.config import IngestConfig, MementoConfig
from memento.gating import GateDecision, PreFilterGate
from memento.memory_store import MemoryStore
from tests.conftest import FakeLLMClient


# ── PreFilterGate (pure unit tests, no DB) ──────────────────────────


def test_prefilter_skips_too_short():
    gate = PreFilterGate(min_chars=20)
    decision = gate.evaluate("hi there")
    assert decision.ingest is False
    assert decision.reason == "too_short"


def test_prefilter_skips_acknowledgments():
    gate = PreFilterGate(min_chars=1)
    for ack in [
        "ok",
        "Okay",
        "yes!",
        "yeah",
        "thanks",
        "Thank you.",
        "Got it",
        "sounds good",
        "no",
        "lol",
        "hi",
    ]:
        decision = gate.evaluate(ack)
        assert decision.ingest is False, f"expected to skip {ack!r}"
        assert decision.reason == "acknowledgment"


def test_prefilter_passes_substantive_content():
    gate = PreFilterGate(min_chars=20)
    decision = gate.evaluate(
        "I just got promoted to Director of Engineering at Acme."
    )
    assert decision.ingest is True
    assert decision.reason == "passed"


def test_prefilter_does_not_swallow_ack_prefixed_content():
    """'ok we should switch the database to postgres' is real content
    even though it starts with 'ok'. The pattern must anchor end-to-end."""
    gate = PreFilterGate(min_chars=1)
    decision = gate.evaluate("ok we should switch the database to postgres")
    assert decision.ingest is True


def test_prefilter_reports_ack_reason_even_when_short():
    """A short ack word should be reported as 'acknowledgment',
    not 'too_short' — the more specific reason is more useful."""
    gate = PreFilterGate(min_chars=50)
    decision = gate.evaluate("ok")
    assert decision.ingest is False
    assert decision.reason == "acknowledgment"


# ── End-to-end gate behavior through MemoryStore ────────────────────


class _FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(8, dtype=np.float32)
        for word in set(text.lower().split()):
            vec[hash(word) % 8] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


def _make_store(*, gate_enabled: bool, store_verbatim_on_skip: bool = True):
    config = MementoConfig(
        db_path=":memory:",
        ingest=IngestConfig(
            gate_enabled=gate_enabled,
            gate_min_chars=20,
            gate_store_verbatim_on_skip=store_verbatim_on_skip,
        ),
    )
    with (
        patch("memento.memory_store.create_embedder", return_value=_FakeEmbedder()),
        patch("memento.llm.create_llm_client", return_value=FakeLLMClient()),
    ):
        return MemoryStore(config)


def test_gate_disabled_by_default():
    store = _make_store(gate_enabled=False)
    try:
        result = store.ingest("ok")
        assert result.gated_out is False
    finally:
        store.close()


def test_gate_skips_acknowledgment_when_enabled():
    store = _make_store(gate_enabled=True)
    try:
        result = store.ingest("ok")
        assert result.gated_out is True
        assert result.gate_reason == "acknowledgment"
        # verbatim still stored by default
        assert result.verbatim_chunk_id != ""
    finally:
        store.close()


def test_gate_skip_can_drop_verbatim():
    store = _make_store(gate_enabled=True, store_verbatim_on_skip=False)
    try:
        result = store.ingest("ok")
        assert result.gated_out is True
        assert result.verbatim_chunk_id == ""
    finally:
        store.close()


def test_bypass_gate_forces_extraction():
    store = _make_store(gate_enabled=True)
    try:
        result = store.ingest("ok", bypass_gate=True)
        assert result.gated_out is False
    finally:
        store.close()


def test_gate_passes_substantive_content():
    store = _make_store(gate_enabled=True)
    try:
        result = store.ingest(
            "I had lunch with Maria from the platform team and we agreed "
            "to ship the new auth service by Friday."
        )
        assert result.gated_out is False
    finally:
        store.close()
