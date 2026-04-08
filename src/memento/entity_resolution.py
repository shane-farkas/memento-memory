"""Entity resolution: match extracted mentions to existing graph entities.

Tier 1 (this file): Name-based signals only (exact, fuzzy, phonetic, alias, type).
Tier 2 (Stage 9): Adds embedding similarity, graph proximity, LLM tiebreaker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from memento.extraction import ExtractedEntity
from memento.graph_store import GraphStore
from memento.models import Entity, EntityType

logger = logging.getLogger(__name__)

# ── Resolution thresholds ────────────────────────────────────────

HIGH_THRESHOLD = 0.85  # Auto-merge
LOW_THRESHOLD = 0.40   # Definitely new entity
# Between 0.40 and 0.85 = ambiguous (Tier 1: conservatively creates new entity)


@dataclass
class ResolutionSignals:
    """Scores from individual resolution signals."""

    name_exact: float | None = None
    name_fuzzy: float | None = None
    name_phonetic: float | None = None
    alias_match: float | None = None
    type_match: float | None = None


@dataclass
class Resolution:
    """The result of resolving a mention against the graph."""

    action: str  # "merge", "create"
    entity: Entity | None = None
    confidence: float = 0.0
    add_alias: str | None = None
    signals: ResolutionSignals = field(default_factory=ResolutionSignals)


# ── String similarity utilities ──────────────────────────────────


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def normalized_edit_similarity(s1: str, s2: str) -> float:
    """Normalized edit similarity (1.0 = identical, 0.0 = completely different)."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (dist / max_len)


def soundex(name: str) -> str:
    """Compute Soundex phonetic code for a name."""
    if not name:
        return ""

    name = name.upper().strip()
    # Keep first letter
    code = name[0]

    # Soundex coding table
    coding = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    prev = coding.get(name[0], "0")
    for char in name[1:]:
        digit = coding.get(char, "0")
        if digit != "0" and digit != prev:
            code += digit
        prev = digit if digit != "0" else prev

    # Pad or truncate to 4 characters
    code = (code + "000")[:4]
    return code


def phonetic_match(name1: str, name2: str) -> float:
    """Check if two names have matching Soundex codes. Returns 0.0 or 1.0."""
    # Compare soundex of each word
    words1 = name1.strip().split()
    words2 = name2.strip().split()

    if not words1 or not words2:
        return 0.0

    # Check if any word pair matches phonetically
    matches = 0
    comparisons = 0
    for w1 in words1:
        for w2 in words2:
            if len(w1) > 1 and len(w2) > 1:  # Skip single letters
                comparisons += 1
                if soundex(w1) == soundex(w2):
                    matches += 1

    if comparisons == 0:
        return 0.0
    return matches / comparisons


# ── Entity Resolver ──────────────────────────────────────────────


class EntityResolver:
    """Resolves extracted entity mentions to existing graph entities.

    Tier 1: Uses only text-based signals (exact name, fuzzy, phonetic, alias, type).
    """

    def __init__(
        self,
        graph: GraphStore,
        high_threshold: float = HIGH_THRESHOLD,
        low_threshold: float = LOW_THRESHOLD,
    ) -> None:
        self.graph = graph
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def resolve(self, mention: ExtractedEntity) -> Resolution:
        """Resolve a single mention against the graph.

        Returns a Resolution indicating whether to merge with an existing entity
        or create a new one.
        """
        # Stage 1: Generate candidates
        candidates = self._generate_candidates(mention)

        if not candidates:
            return Resolution(action="create")

        # Stage 2: Score each candidate
        scored = []
        for candidate in candidates:
            signals = self._score_candidate(mention, candidate)
            score = self._compute_composite_score(signals)
            scored.append((candidate, score, signals))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best_candidate, best_score, best_signals = scored[0]

        # Stage 3: Decision
        if best_score >= self.high_threshold:
            add_alias = (
                mention.name
                if mention.name != best_candidate.name
                and mention.name not in best_candidate.aliases
                else None
            )
            return Resolution(
                action="merge",
                entity=best_candidate,
                confidence=best_score,
                add_alias=add_alias,
                signals=best_signals,
            )
        elif best_score >= self.low_threshold:
            # Tier 1: conservatively creates new entity in the ambiguous zone.
            # Tier 2 will add LLM tiebreaker here.
            logger.debug(
                "Ambiguous resolution for '%s' → '%s' (score: %.2f). "
                "Creating new entity (conservative Tier 1 behavior).",
                mention.name,
                best_candidate.name,
                best_score,
            )
            return Resolution(action="create")
        else:
            return Resolution(action="create")

    def resolve_batch(
        self, mentions: list[ExtractedEntity]
    ) -> list[tuple[ExtractedEntity, Resolution]]:
        """Resolve a batch of mentions. Returns (mention, resolution) pairs."""
        return [(m, self.resolve(m)) for m in mentions]

    # ── Stage 1: Candidate Generation ────────────────────────────

    def _generate_candidates(self, mention: ExtractedEntity) -> list[Entity]:
        """Generate candidate entities that might match this mention."""
        candidates = {}

        # Exact name match (including aliases)
        exact = self.graph.search_entities(name=mention.name, fuzzy=False)
        for e in exact:
            candidates[e.id] = e

        # Fuzzy name match
        fuzzy = self.graph.search_entities(name=mention.name, fuzzy=True)
        for e in fuzzy:
            candidates[e.id] = e

        # If mention has multiple words, try each word as a search
        words = mention.name.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 2:  # Skip short words
                    partial = self.graph.search_entities(name=word, fuzzy=True)
                    for e in partial:
                        candidates[e.id] = e

        return list(candidates.values())

    # ── Stage 2: Candidate Scoring ───────────────────────────────

    def _score_candidate(
        self, mention: ExtractedEntity, candidate: Entity
    ) -> ResolutionSignals:
        """Score a candidate on text-based signals."""
        signals = ResolutionSignals()

        # Exact name match
        if mention.name.lower() == candidate.name.lower():
            signals.name_exact = 1.0
        else:
            signals.name_exact = 0.0

        # Fuzzy name match (normalized edit similarity)
        signals.name_fuzzy = normalized_edit_similarity(mention.name, candidate.name)

        # Phonetic match
        signals.name_phonetic = phonetic_match(mention.name, candidate.name)

        # Alias match (None if no aliases exist — signal is unavailable)
        if candidate.aliases:
            alias_scores = []
            for alias in candidate.aliases:
                if mention.name.lower() == alias.lower():
                    alias_scores.append(1.0)
                else:
                    alias_scores.append(
                        normalized_edit_similarity(mention.name, alias)
                    )
            signals.alias_match = max(alias_scores)
        else:
            signals.alias_match = None  # No aliases → signal unavailable

        # Type match
        if mention.type == candidate.type:
            signals.type_match = 1.0
        elif _types_compatible(mention.type, candidate.type):
            signals.type_match = 0.5
        else:
            signals.type_match = 0.0

        return signals

    def _compute_composite_score(self, signals: ResolutionSignals) -> float:
        """Compute a weighted composite score from individual signals.

        Uses fast-path logic for clear matches, and falls back to weighted
        scoring for ambiguous cases.
        """
        type_ok = signals.type_match is not None and signals.type_match >= 0.5

        # Fast path: exact name match with compatible type → very high confidence
        if signals.name_exact == 1.0 and type_ok:
            return 0.95

        # Fast path: exact alias match with compatible type → high confidence
        if signals.alias_match is not None and signals.alias_match == 1.0 and type_ok:
            return 0.92

        # Weighted scoring for fuzzy/ambiguous cases
        # Only include signals that are available and non-binary-zero
        weights = {
            "name_fuzzy": 0.35,
            "name_phonetic": 0.10,
            "alias_match": 0.25,
            "type_match": 0.30,
        }

        available = {}
        for key, weight in weights.items():
            val = getattr(signals, key)
            if val is not None:
                available[key] = (val, weight)

        if not available:
            return 0.0

        total_weight = sum(w for _, w in available.values())
        score = sum(v * w for v, w in available.values())
        return score / total_weight if total_weight > 0 else 0.0


def _types_compatible(t1: EntityType, t2: EntityType) -> bool:
    """Check if two entity types are compatible (not contradictory)."""
    # Same type is always compatible
    if t1 == t2:
        return True

    # Some types are never compatible
    incompatible_pairs = {
        frozenset({EntityType.PERSON, EntityType.ORGANIZATION}),
        frozenset({EntityType.PERSON, EntityType.LOCATION}),
        frozenset({EntityType.ORGANIZATION, EntityType.LOCATION}),
    }
    return frozenset({t1, t2}) not in incompatible_pairs


# ── Tier 2: Full Multi-Signal Resolution ─────────────────────────


@dataclass
class Tier2Signals(ResolutionSignals):
    """Extended signals including embeddings, graph proximity, and context."""

    description_sim: float | None = None
    context_sim: float | None = None
    shared_neighbors: float | None = None
    recency: float | None = None


class Tier2EntityResolver(EntityResolver):
    """Full multi-signal entity resolution with embedding similarity,
    graph proximity, and LLM tiebreaker for ambiguous cases.

    Extends Tier 1 with:
    - Embedding similarity (description + context)
    - Graph structural signals (shared neighbors)
    - LLM tiebreaker for the ambiguous zone
    - Cold-start weight schedule
    """

    def __init__(
        self,
        graph: GraphStore,
        embedder=None,
        tiebreaker_model: str = "",
        llm_client=None,
        high_threshold: float = HIGH_THRESHOLD,
        low_threshold: float = LOW_THRESHOLD,
    ) -> None:
        super().__init__(graph, high_threshold, low_threshold)
        self.embedder = embedder
        self.tiebreaker_model = tiebreaker_model
        self._llm = llm_client

    def resolve(self, mention: ExtractedEntity) -> Resolution:
        """Resolve with full multi-signal scoring + LLM tiebreaker."""
        candidates = self._generate_candidates(mention)
        if not candidates:
            return Resolution(action="create")

        scored = []
        for candidate in candidates:
            signals = self._score_candidate_tier2(mention, candidate)
            score = self._compute_tier2_score(signals)
            scored.append((candidate, score, signals))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_candidate, best_score, best_signals = scored[0]

        # Apply disqualifiers
        best_score = self._apply_disqualifiers(mention, best_candidate, best_score)

        if best_score >= self.high_threshold:
            add_alias = (
                mention.name
                if mention.name != best_candidate.name
                and mention.name not in best_candidate.aliases
                else None
            )
            return Resolution(
                action="merge",
                entity=best_candidate,
                confidence=best_score,
                add_alias=add_alias,
                signals=best_signals,
            )
        elif best_score >= self.low_threshold:
            # Tier 2: LLM tiebreaker for the ambiguous zone
            llm_result = self._llm_tiebreaker(mention, best_candidate)
            if llm_result:
                return Resolution(
                    action="merge",
                    entity=best_candidate,
                    confidence=best_score * 0.95,
                    add_alias=mention.name if mention.name != best_candidate.name else None,
                    signals=best_signals,
                )
            return Resolution(action="create")
        else:
            return Resolution(action="create")

    def _score_candidate_tier2(
        self, mention: ExtractedEntity, candidate: Entity
    ) -> Tier2Signals:
        """Score with all signals including embeddings and graph structure."""
        # Start with Tier 1 signals
        base = self._score_candidate(mention, candidate)
        signals = Tier2Signals(
            name_exact=base.name_exact,
            name_fuzzy=base.name_fuzzy,
            name_phonetic=base.name_phonetic,
            alias_match=base.alias_match,
            type_match=base.type_match,
        )

        # Embedding similarity
        if self.embedder is not None:
            mention_text = f"{mention.name} ({mention.type.value})"
            candidate_text = f"{candidate.name} ({candidate.type.value})"
            try:
                m_emb = self.embedder.embed(mention_text)
                c_emb = self.embedder.embed(candidate_text)
                import numpy as np
                sim = float(np.dot(m_emb, c_emb) / (
                    np.linalg.norm(m_emb) * np.linalg.norm(c_emb) + 1e-8
                ))
                signals.description_sim = max(0.0, sim)
            except Exception:
                pass

        # Graph structural: shared neighbors
        try:
            mention_neighbors = set()  # Mention has no neighbors yet
            candidate_neighbors = {
                e.id for e in self.graph.get_neighbors(candidate.id, max_hops=1)
            }
            # If the mention co-occurs with entities that are neighbors of the candidate,
            # that's a strong signal. For now, just measure candidate connectivity.
            if candidate_neighbors:
                signals.shared_neighbors = min(len(candidate_neighbors) / 10.0, 1.0)
            else:
                signals.shared_neighbors = 0.0
        except Exception:
            pass

        # Recency: how recently was this candidate active?
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            last_seen = datetime.fromisoformat(candidate.last_seen)
            days_since = (now - last_seen).days
            import math
            signals.recency = math.exp(-days_since / 30.0)
        except Exception:
            signals.recency = 0.5

        return signals

    def _compute_tier2_score(self, signals: Tier2Signals) -> float:
        """Compute composite score with cold-start weight schedule."""
        type_ok = signals.type_match is not None and signals.type_match >= 0.5

        # Fast paths (same as Tier 1)
        if signals.name_exact == 1.0 and type_ok:
            return 0.95
        if signals.alias_match is not None and signals.alias_match == 1.0 and type_ok:
            return 0.92

        # Get adaptive weights based on graph density
        weights = self._get_adaptive_weights()

        available = {}
        for key, weight in weights.items():
            val = getattr(signals, key, None)
            if val is not None:
                available[key] = (val, weight)

        if not available:
            return 0.0

        total_weight = sum(w for _, w in available.values())
        score = sum(v * w for v, w in available.values())
        return score / total_weight if total_weight > 0 else 0.0

    def _get_adaptive_weights(self) -> dict[str, float]:
        """Cold-start weight schedule: shift from text to structural signals
        as the graph densifies."""
        stats = self.graph.stats()
        node_count = stats["node_count"]
        edge_count = stats["edge_count"]
        density = edge_count / max(node_count, 1)
        maturity = min(density / 3.0, 1.0)

        def lerp(start: float, end: float) -> float:
            return start + (end - start) * maturity

        return {
            "name_fuzzy": lerp(0.35, 0.25),
            "name_phonetic": lerp(0.08, 0.05),
            "alias_match": lerp(0.25, 0.20),
            "type_match": lerp(0.20, 0.15),
            "description_sim": lerp(0.05, 0.10),
            "shared_neighbors": lerp(0.02, 0.12),
            "recency": lerp(0.05, 0.08),
        }

    def _apply_disqualifiers(
        self, mention: ExtractedEntity, candidate: Entity, score: float
    ) -> float:
        """Apply hard disqualifiers that override the score."""
        # Type incompatibility
        if mention.type and candidate.type:
            if not _types_compatible(mention.type, candidate.type):
                return 0.0
        return score

    def _llm_tiebreaker(
        self, mention: ExtractedEntity, candidate: Entity
    ) -> bool:
        """Use LLM to decide if an ambiguous match is the same entity.

        Returns True if the LLM thinks they're the same entity.
        """
        if self._llm is None:
            return False

        try:
            props = candidate.properties or {}
            facts = ", ".join(f"{k}: {v.value}" for k, v in props.items()) or "no known facts"

            prompt = f"""I have an existing entity in my knowledge base:
  Name: {candidate.name}
  Aliases: {', '.join(candidate.aliases) if candidate.aliases else 'none'}
  Type: {candidate.type.value}
  Known facts: {facts}

A new mention appeared in conversation:
  Text: "{mention.name}"
  Type hint: {mention.type.value}

Are these the same entity?
Reply with exactly one word: YES or NO"""

            answer = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.tiebreaker_model,
                temperature=0.0,
                max_tokens=10,
            )
            return answer.strip().upper().startswith("YES")
        except Exception as e:
            logger.warning("LLM tiebreaker failed: %s", e)
            return False
