"""Full retrieval engine: graph traversal, composite ranking, context budgeting.

Implements the 6-step retrieval pipeline from the architecture doc:
1. Entity identification from query
2. Subgraph expansion
3. Temporal filtering
4. Relevance ranking (7 signals)
5. Context budgeting with coherence pass
6. Serialization
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from memento.graph_store import GraphStore
from memento.models import Entity, EntityType, PropertyValue, Relationship
from memento.reranker import Reranker, sigmoid
from memento.verbatim_store import VerbatimStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFact:
    """A single fact (property or relationship) in the candidate set."""

    entity: Entity
    fact_type: str  # "property" or "relationship"
    key: str = ""
    value: str = ""
    target_entity: Entity | None = None
    confidence: float = 1.0
    recorded_at: str = ""
    source_authority: float = 1.0
    relationship: Relationship | None = None

    def natural_language(self) -> str:
        if self.fact_type == "property":
            return f"{self.key}: {self.value}"
        elif self.fact_type == "relationship" and self.target_entity:
            return f"{self.key} {self.target_entity.name}"
        return self.value


@dataclass
class MemoryContext:
    """The assembled retrieval result ready for LLM consumption."""

    text: str
    facts: list[RetrievalFact] = field(default_factory=list)
    entity_count: int = 0
    token_estimate: int = 0


class RetrievalEngine:
    """Full retrieval pipeline with graph traversal and composite ranking."""

    def __init__(
        self,
        graph: GraphStore,
        verbatim: VerbatimStore | None = None,
        default_token_budget: int = 2000,
        max_hop_depth: int = 3,
        reranker: Reranker | None = None,
        recency_decay_days: float = 30.0,
        semantic_entity_recall: bool = True,
        semantic_entity_top_k: int = 8,
    ) -> None:
        self.graph = graph
        self.verbatim = verbatim
        self.default_token_budget = default_token_budget
        self.max_hop_depth = max_hop_depth
        self.reranker = reranker
        self.recency_decay_days = recency_decay_days
        self.semantic_entity_recall = semantic_entity_recall
        self.semantic_entity_top_k = semantic_entity_top_k

    def recall(
        self,
        query: str,
        context: str = "",
        token_budget: int | None = None,
        as_of: str | None = None,
    ) -> MemoryContext:
        """Execute the full retrieval pipeline."""
        budget = token_budget or self.default_token_budget

        # Step 0: Adaptive retrieval — widen search for enumeration queries
        retrieval_profile = self._classify_query(query)
        verbatim_top_k = retrieval_profile["verbatim_top_k"]
        max_conversations = retrieval_profile["max_conversations"]

        # Step 1: Identify entities in the query
        query_entities = self._identify_entities(query)

        if not query_entities and not self.verbatim:
            return MemoryContext(text="No relevant memories found.")

        # Step 2: Expand subgraph from identified entities
        candidate_facts = self._expand_subgraph(query_entities, as_of)

        # Step 3: Temporal filtering
        if not as_of:
            candidate_facts = self._temporal_filter(candidate_facts)

        # Step 4: Rank all candidate facts
        ranked = self._rank_facts(candidate_facts, query_entities, query, as_of)

        # Step 5: Context budgeting with coherence pass
        selected = self._budget_and_select(ranked, budget, query_entities)

        # Step 6: Blend with verbatim results if available
        verbatim_text = ""
        if self.verbatim:
            verbatim_results = self.verbatim.search(query, top_k=verbatim_top_k)
            # Cross-encoder rerank: reorder chunks by joint query-document
            # relevance so the strongest conversations are pulled first.
            if verbatim_results and self.reranker:
                verbatim_results = self._rerank_results(query, verbatim_results)
            if verbatim_results:
                # Group by conversation and expand: if a single turn matched,
                # pull in the full session so the LLM has surrounding context.
                seen_convs: dict[str, list] = {}
                for r in verbatim_results:
                    cid = r.conversation_id
                    if cid and cid not in seen_convs:
                        session_chunks = self.verbatim.get_by_conversation(cid)
                        seen_convs[cid] = session_chunks
                    elif not cid and r.chunk_id not in seen_convs:
                        seen_convs[r.chunk_id] = [r]

                verbatim_lines = ["## Related Conversations"]
                for cid, chunks in list(seen_convs.items())[:max_conversations]:
                    ts = chunks[0].timestamp if chunks else ""
                    ts_label = f" (recorded: {ts})" if ts else ""
                    full_text = "\n".join(c.text for c in chunks)
                    verbatim_lines.append(
                        f"### Conversation{ts_label}\n{full_text}"
                    )
                verbatim_text = "\n\n".join(verbatim_lines)

        # Step 7: Serialize
        text = self._serialize(selected, query_entities, verbatim_text)
        return MemoryContext(
            text=text,
            facts=selected,
            entity_count=len({f.entity.id for f in selected}),
            token_estimate=len(text) // 4,
        )

    # ── Step 1: Entity Identification ────────────────────────────

    def _identify_entities(self, query: str) -> list[Entity]:
        """Find entities in the query by searching the graph."""
        entities = []
        seen_ids = set()

        # Try the full query as an entity name
        results = self.graph.search_entities(name=query, fuzzy=True)
        for e in results:
            if e.id not in seen_ids:
                entities.append(e)
                seen_ids.add(e.id)

        # Try individual words (for multi-word queries)
        words = query.split()
        for word in words:
            if len(word) > 2:
                results = self.graph.search_entities(name=word, fuzzy=True)
                for e in results:
                    if e.id not in seen_ids:
                        entities.append(e)
                        seen_ids.add(e.id)

        # Semantic recall: the query may not contain any entity *name* string
        # (e.g. "where did I say I wanted to travel?"). Seed the graph from
        # entities that appear in semantically-retrieved verbatim chunks so the
        # graph is reachable beyond literal name overlap.
        if self.semantic_entity_recall and self.verbatim:
            for e in self._semantic_entity_recall(query, seen_ids):
                entities.append(e)
                seen_ids.add(e.id)

        return entities

    def _semantic_entity_recall(
        self, query: str, seen_ids: set[str]
    ) -> list[Entity]:
        """Find entities mentioned in the top semantically-similar chunks.

        Bridges the dense-retrieval channel into the graph: we vector-search
        the verbatim store, then match known entity names/aliases against the
        retrieved text. This recovers the relevant subgraph for queries that
        share no surface tokens with any entity name.
        """
        try:
            chunks = self.verbatim._vector_search(query, self.semantic_entity_top_k)
        except Exception:
            return []
        if not chunks:
            return []

        blob = "\n".join(c.text.lower() for c in chunks)
        # Token set for a cheap reject before the (more expensive) whole-word
        # phrase check — avoids running a regex over the blob for every entity.
        blob_words = set(re.findall(r"[a-z0-9]+", blob))

        found: list[Entity] = []
        for entity in self.graph.search_entities():
            if entity.id in seen_ids:
                continue
            names = [entity.name, *(entity.aliases or [])]
            if any(self._name_in_blob(n, blob, blob_words) for n in names):
                found.append(entity)
                seen_ids.add(entity.id)
                # Bound the scan: stop once we've recalled enough fresh entities
                # so a large graph can't turn every recall into a full scan.
                if len(found) >= self.semantic_entity_top_k:
                    break
        return found

    @staticmethod
    def _name_in_blob(name: str | None, blob: str, blob_words: set[str]) -> bool:
        """True if *name* occurs in *blob* as a whole word/phrase, not a substring.

        Substring matching produced false positives ("Sam" inside "sample",
        "Ana" inside "banana"). We require every token of the name to be present
        (cheap set check) and then confirm the full name as a word-bounded
        phrase, so multi-word names like "Alpha Corp" still match.
        """
        if not name or len(name) < 3:
            return False
        nl = name.lower()
        name_words = re.findall(r"[a-z0-9]+", nl)
        if not name_words or not all(w in blob_words for w in name_words):
            return False
        return re.search(r"\b" + re.escape(nl) + r"\b", blob) is not None

    def _rerank_results(self, query, results):
        """Reorder verbatim results by cross-encoder relevance (descending)."""
        docs = [r.text for r in results]
        try:
            scores = self.reranker.score(query, docs)
        except Exception:
            return results
        if len(scores) != len(results):
            return results
        for r, s in zip(results, scores):
            r.score = s
        return sorted(results, key=lambda r: r.score, reverse=True)

    # ── Step 2: Subgraph Expansion ───────────────────────────────

    def _expand_subgraph(
        self, query_entities: list[Entity], as_of: str | None = None
    ) -> list[RetrievalFact]:
        """Walk the graph from query entities, collecting candidate facts."""
        facts = []
        seen_fact_ids = set()

        for qe in query_entities:
            # Get the entity with its properties
            if as_of:
                entity = self.graph.point_in_time_snapshot(qe.id, as_of)
            else:
                entity = self.graph.get_entity(qe.id)

            if not entity:
                continue

            # Always add a basic fact for the entity itself (ensures it appears)
            base_id = f"entity:{entity.id}"
            if base_id not in seen_fact_ids:
                facts.append(RetrievalFact(
                    entity=entity,
                    fact_type="property",
                    key="type",
                    value=entity.type.value,
                    confidence=entity.confidence,
                    recorded_at=entity.created_at,
                ))
                seen_fact_ids.add(base_id)

            # Add property facts
            for key, prop in entity.properties.items():
                fact_id = f"prop:{entity.id}:{key}"
                if fact_id not in seen_fact_ids:
                    facts.append(RetrievalFact(
                        entity=entity,
                        fact_type="property",
                        key=key,
                        value=str(prop.value),
                        confidence=prop.confidence,
                        recorded_at=prop.recorded_at,
                    ))
                    seen_fact_ids.add(fact_id)

            # Add relationship facts (1-hop)
            rels = self.graph.get_relationships(entity.id)
            for rel in rels:
                fact_id = f"rel:{rel.id}"
                if fact_id in seen_fact_ids:
                    continue

                other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
                other = self.graph.get_entity(other_id)
                if not other:
                    continue

                direction = "→" if rel.source_id == entity.id else "←"
                facts.append(RetrievalFact(
                    entity=entity,
                    fact_type="relationship",
                    key=f"{direction} [{rel.type}]",
                    value=other.name,
                    target_entity=other,
                    confidence=rel.confidence,
                    recorded_at=rel.valid_from,
                    relationship=rel,
                ))
                seen_fact_ids.add(fact_id)

                # 2-hop: also add properties of connected entities
                for key, prop in (other.properties or {}).items():
                    hop2_id = f"prop:{other.id}:{key}"
                    if hop2_id not in seen_fact_ids:
                        facts.append(RetrievalFact(
                            entity=other,
                            fact_type="property",
                            key=key,
                            value=str(prop.value),
                            confidence=prop.confidence * 0.8,  # Discount for distance
                            recorded_at=prop.recorded_at,
                        ))
                        seen_fact_ids.add(hop2_id)

        return facts

    # ── Step 3: Temporal Filter ──────────────────────────────────

    def _temporal_filter(self, facts: list[RetrievalFact]) -> list[RetrievalFact]:
        """Remove superseded or expired facts for current-state queries."""
        # For now, rely on the graph store returning only current properties
        # (superseded values are excluded by default in get_entity)
        return [f for f in facts if f.confidence > 0.05]

    # ── Step 4: Relevance Ranking ────────────────────────────────

    def _rank_facts(
        self,
        facts: list[RetrievalFact],
        query_entities: list[Entity],
        query: str,
        as_of: str | None = None,
    ) -> list[tuple[RetrievalFact, float]]:
        """Score and rank facts using composite signals.

        When a reranker is configured, each fact is also scored by the
        cross-encoder against the query (batched, normalized to (0,1)) and that
        signal is blended into the composite — putting graph facts and verbatim
        chunks on a comparable relevance scale.
        """
        query_entity_ids = {e.id for e in query_entities}

        # Keyed by list index, not id(fact): object identity is fragile (an id
        # can be reused once a fact is GC'd) and index keying is just as cheap.
        rerank_scores: dict[int, float] = {}
        if self.reranker and facts:
            docs = [self._fact_doc(f) for f in facts]
            try:
                raw = self.reranker.score(query, docs)
                if len(raw) == len(facts):
                    rerank_scores = {i: sigmoid(s) for i, s in enumerate(raw)}
            except Exception:
                rerank_scores = {}

        scored = []
        for i, fact in enumerate(facts):
            score = self._score_fact(
                fact, query_entity_ids, query, as_of,
                rerank=rerank_scores.get(i),
            )
            scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _fact_doc(self, fact: RetrievalFact) -> str:
        """Render a fact as a short document for the cross-encoder."""
        return f"{fact.entity.name}: {fact.natural_language()}"

    def _score_fact(
        self,
        fact: RetrievalFact,
        query_entity_ids: set[str],
        query: str,
        as_of: str | None = None,
        rerank: float | None = None,
    ) -> float:
        """Composite ranking: relevance, recency, confidence, authority."""
        # Signal 1: Query relevance (is this about a query entity?)
        if fact.entity.id in query_entity_ids:
            query_relevance = 1.0
        elif fact.target_entity and fact.target_entity.id in query_entity_ids:
            query_relevance = 0.8
        else:
            query_relevance = 0.3

        # Signal 2: Recency — measured relative to the query's as_of date, not
        # wall-clock now, so it stays meaningful when reasoning over a past
        # point in time (the entire LongMemEval setting).
        recency = self._compute_recency(fact, as_of)

        # Signal 3: Confidence
        confidence = fact.confidence

        # Signal 4: Source authority
        authority = fact.source_authority

        # Composite: multiplicative gate + additive rank. When a cross-encoder
        # relevance signal is available it replaces a chunk of the heuristic
        # query-relevance/recency weight with a far sharper relevance estimate.
        gate = confidence * authority
        if rerank is not None:
            rank = (
                0.30 * query_relevance
                + 0.20 * recency
                + 0.20 * confidence
                + 0.30 * rerank
            )
        else:
            rank = (
                0.40 * query_relevance
                + 0.30 * recency
                + 0.30 * confidence
            )
        return gate * rank

    def _parse_ref_time(self, as_of: str | None) -> datetime:
        """Reference time for recency decay: the as_of date, else now (UTC)."""
        if as_of:
            try:
                dt = datetime.fromisoformat(as_of)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
        return datetime.now(timezone.utc)

    def _compute_recency(
        self, fact: RetrievalFact, as_of: str | None = None
    ) -> float:
        """Compute recency score (exponential decay) relative to as_of/now."""
        if not fact.recorded_at:
            return 0.5
        try:
            recorded = datetime.fromisoformat(fact.recorded_at)
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=timezone.utc)
            ref = self._parse_ref_time(as_of)
            days_since = max(0, (ref - recorded).days)
            return math.exp(-days_since / self.recency_decay_days)
        except Exception:
            return 0.5

    # ── Step 5: Context Budgeting ────────────────────────────────

    def _budget_and_select(
        self,
        ranked: list[tuple[RetrievalFact, float]],
        token_budget: int,
        query_entities: list[Entity],
    ) -> list[RetrievalFact]:
        """Greedily select facts within token budget with coherence pass."""
        selected = []
        remaining_budget = token_budget
        selected_entities = set()

        # Phase 1: Seed with facts about query entities
        for fact, score in ranked:
            if fact.entity.id in {e.id for e in query_entities} and score > 0.3:
                cost = self._estimate_tokens(fact)
                if cost <= remaining_budget:
                    selected.append(fact)
                    remaining_budget -= cost
                    selected_entities.add(fact.entity.id)
                    if fact.target_entity:
                        selected_entities.add(fact.target_entity.id)

        # Phase 2: Add remaining high-value facts
        for fact, score in ranked:
            if fact in selected:
                continue
            if score < 0.1:
                break
            cost = self._estimate_tokens(fact)
            if cost <= remaining_budget:
                selected.append(fact)
                remaining_budget -= cost
                selected_entities.add(fact.entity.id)

        # Phase 3: Coherence pass — add introductions for referenced entities
        referenced = set()
        for fact in selected:
            if fact.target_entity:
                referenced.add(fact.target_entity.id)

        for eid in referenced - selected_entities:
            entity = self.graph.get_entity(eid)
            if entity and remaining_budget > 20:
                intro_fact = RetrievalFact(
                    entity=entity,
                    fact_type="property",
                    key="type",
                    value=entity.type.value,
                    confidence=1.0,
                )
                cost = self._estimate_tokens(intro_fact)
                if cost <= remaining_budget:
                    selected.append(intro_fact)
                    remaining_budget -= cost

        return selected

    def _estimate_tokens(self, fact: RetrievalFact) -> int:
        """Estimate token count for a fact (4 chars ≈ 1 token)."""
        text = fact.natural_language()
        return max(1, len(text) // 4 + 5)  # +5 for formatting overhead

    # ── Step 6: Serialization ────────────────────────────────────

    def _serialize(
        self,
        facts: list[RetrievalFact],
        query_entities: list[Entity],
        verbatim_text: str = "",
    ) -> str:
        """Serialize selected facts into natural language for the LLM."""
        if not facts and not verbatim_text:
            return "No relevant memories found."

        # Group facts by entity
        entity_groups: dict[str, list[RetrievalFact]] = {}
        entity_map: dict[str, Entity] = {}
        for fact in facts:
            eid = fact.entity.id
            if eid not in entity_groups:
                entity_groups[eid] = []
                entity_map[eid] = fact.entity
            entity_groups[eid].append(fact)

        # Serialize each entity group
        output = []
        for eid, group in entity_groups.items():
            entity = entity_map[eid]
            section = [f"## {entity.name} ({entity.type.value})"]

            # Properties first, then relationships
            props = [f for f in group if f.fact_type == "property"]
            rels = [f for f in group if f.fact_type == "relationship"]

            for fact in props:
                line = f"- {fact.natural_language()}"
                if fact.confidence < 0.6:
                    line += " *(unconfirmed)*"
                section.append(line)

            for fact in rels:
                line = f"- {fact.natural_language()}"
                if fact.confidence < 0.6:
                    line += " *(unconfirmed)*"
                section.append(line)

            output.append("\n".join(section))

        if verbatim_text:
            output.append(verbatim_text)

        return "\n\n".join(output)

    def _is_simple_recall(self, query: str) -> bool:
        """Heuristic: is this a simple factual recall vs. compositional query?"""
        simple_indicators = ["what did", "when did", "what was", "did i", "have i"]
        query_lower = query.lower()
        return any(ind in query_lower for ind in simple_indicators)

    def _classify_query(self, query: str) -> dict:
        """Classify a query to select retrieval parameters.

        Wide-recall queries (counting, listing, multi-hop) get more
        verbatim results and conversations. Narrow queries (single fact,
        simple recall) stay focused to avoid context dilution.
        """
        q = query.lower()

        wide_patterns = [
            "how many", "how much", "list all", "list every",
            "all the", "everything", "total number",
            "did i participate", "did i attend", "did i visit",
        ]
        is_wide = any(p in q for p in wide_patterns)

        if is_wide:
            return {"verbatim_top_k": 20, "max_conversations": 10}
        return {"verbatim_top_k": 10, "max_conversations": 5}
