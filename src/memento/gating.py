"""Ingest gating: cheap pre-filters that decide whether text deserves the full pipeline.

This is Phase 1 of the signal-detector design — only deterministic checks
(length, acknowledgment patterns). An LLM-backed gate (Phase 3) can plug
into the same `Gate` protocol later without changing the call site.

The gate is biased toward ingest: only short or clearly content-free turns
are skipped. False negatives (missed memories) are the costly error mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


_ACKNOWLEDGMENT_PATTERN = re.compile(
    r"^[\s\W]*("
    r"ok(?:ay)?|yes|yeah|yep|yup|sure|sounds good|got it|"
    r"thanks|thank you|ty|thx|"
    r"no|nope|nah|"
    r"cool|nice|great|awesome|perfect|"
    r"k|kk|ack|"
    r"hi|hello|hey|"
    r"bye|goodbye|cya|"
    r"lol|lmao|haha"
    r")[\s\W]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GateDecision:
    """Result of evaluating whether a piece of text should be ingested."""

    ingest: bool
    reason: str


class Gate(Protocol):
    """Protocol every ingest gate implements."""

    def evaluate(self, text: str) -> GateDecision: ...


class PreFilterGate:
    """Deterministic, LLM-free gate.

    Skips text that is too short or matches a closed list of
    acknowledgment / small-talk patterns. Everything else passes.
    """

    def __init__(self, min_chars: int = 20) -> None:
        self._min_chars = min_chars

    def evaluate(self, text: str) -> GateDecision:
        stripped = text.strip()
        # Check acknowledgment patterns first so short ack words ("ok", "yes")
        # are reported with the more informative reason rather than "too_short".
        if _ACKNOWLEDGMENT_PATTERN.match(stripped):
            return GateDecision(ingest=False, reason="acknowledgment")
        if len(stripped) < self._min_chars:
            return GateDecision(ingest=False, reason="too_short")
        return GateDecision(ingest=True, reason="passed")
