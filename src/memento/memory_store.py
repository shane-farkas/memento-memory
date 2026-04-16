"""MemoryStore: the public API wrapping all Memento components."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from memento.config import MementoConfig
from memento.conflict import ConflictDetector, ConflictResult, ConflictType
from memento.consolidation import ConsolidationEngine, ConsolidationResult
from memento.db import Database
from memento.embedder import Embedder, create_embedder
from memento.entity_resolution import EntityResolver, Tier2EntityResolver
from memento.extraction import EntityExtractor, RelationExtractor
from memento.gating import Gate, GateDecision, PreFilterGate
from memento.graph_store import GraphStore
from memento.models import (
    Conflict,
    Entity,
    EntityType,
    PropertyValue,
    SourceRef,
    _new_id,
    _now,
)
from memento.privacy import DeletionReceipt, EntityDataExport, BeliefChain, PrivacyLayer
from memento.retrieval import MemoryContext, RetrievalEngine
from memento.scratchpad import SessionScratchpad
from memento.verbatim_store import VerbatimStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting text into the memory system."""

    entities_created: list[str]
    entities_resolved: list[str]
    relationships_created: int
    conflicts_detected: int
    verbatim_chunk_id: str
    gated_out: bool = False
    gate_reason: str = ""


@dataclass
class GraphHealth:
    """Health statistics for the knowledge graph."""

    node_count: int
    edge_count: int
    active_property_count: int
    avg_confidence: float
    unresolved_conflicts: int


class Session:
    """An active conversation session with scratchpad tracking."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._scratchpad = SessionScratchpad(graph=store.graph)
        self._turn = 0

    @property
    def session_id(self) -> str:
        return self._scratchpad.session_id

    def on_turn(self, text: str) -> None:
        """Process a conversation turn through the scratchpad."""
        self._turn += 1
        self._scratchpad.on_turn(text, self._turn)

    def end(self) -> IngestResult | None:
        """End the session: flush scratchpad contents through ingestion."""
        unique = self._scratchpad.get_unique_entities()
        if not unique:
            return None

        # Build a combined text from all scratchpad mentions
        texts = []
        for mention in self._scratchpad.mentions:
            if mention.surrounding_context and mention.surrounding_context not in texts:
                texts.append(mention.surrounding_context)

        combined = " ".join(texts) if texts else ""
        if combined:
            return self._store.ingest(combined, source_type="session")
        return None


class MemoryStore:
    """The public API for the Memento memory system.

    Wraps all components (graph, verbatim, extraction, resolution,
    conflict detection, retrieval) into a single interface.
    """

    def __init__(self, config: MementoConfig | None = None) -> None:
        self.config = config or MementoConfig()
        self._db = Database(self.config.db_path)
        self.graph = GraphStore(self._db)

        self._embedder = create_embedder(self.config.embedding)
        self.verbatim = VerbatimStore(self._db, self._embedder)

        from memento.llm import create_llm_client, get_default_model

        self._llm = create_llm_client(self.config.llm)

        # Resolve model names (use defaults if not configured)
        provider = self.config.llm.provider or "anthropic"
        extraction_model = self.config.llm.extraction_model or get_default_model(provider, "extraction")
        tiebreaker_model = self.config.llm.tiebreaker_model or get_default_model(provider, "extraction")

        self._entity_extractor = EntityExtractor(
            model=extraction_model, llm_client=self._llm
        )
        self._relation_extractor = RelationExtractor(
            model=extraction_model, llm_client=self._llm
        )
        self._resolver = Tier2EntityResolver(
            self.graph,
            embedder=self._embedder,
            tiebreaker_model=tiebreaker_model,
            llm_client=self._llm,
            high_threshold=self.config.resolution.high_threshold,
            low_threshold=self.config.resolution.low_threshold,
        )
        self._conflict_detector = ConflictDetector(self.graph)
        self._retrieval = RetrievalEngine(
            self.graph,
            verbatim=self.verbatim,
            default_token_budget=self.config.retrieval.default_token_budget,
            max_hop_depth=self.config.retrieval.max_hop_depth,
        )
        self._consolidation = ConsolidationEngine(
            self.graph,
            half_lives=self.config.consolidation.half_lives,
        )
        self._privacy = PrivacyLayer(self.graph)
        self._ingestion_count = 0

        self._gate: Gate | None = (
            PreFilterGate(min_chars=self.config.ingest.gate_min_chars)
            if self.config.ingest.gate_enabled
            else None
        )

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        conversation_id: str = "",
        turn_number: int = 0,
        source_type: str = "conversation",
        authority: float = 0.9,
        timestamp: str | None = None,
        bypass_gate: bool = False,
    ) -> IngestResult:
        """Process raw text and update the knowledge graph.

        Runs the full pipeline: verbatim storage, entity extraction,
        entity resolution, conflict detection, and graph writes.

        Args:
            timestamp: Optional ISO-8601 timestamp for historical ingestion.
                       When set, the source reference and property as_of times
                       use this value instead of the current wall-clock time.
            bypass_gate: When True, skip the ingest gate even if it is
                         configured. Use when the caller is certain the
                         content matters (e.g. an explicit "remember this").
        """
        conversation_id = conversation_id or _new_id()

        # Ingest gate: short-circuit on clearly content-free turns. Verbatim
        # storage still happens by default so FTS5 keyword search can recover
        # the text even when extraction was skipped.
        if self._gate is not None and not bypass_gate:
            decision = self._gate.evaluate(text)
            if not decision.ingest:
                gated_chunk_id = ""
                if self.config.ingest.gate_store_verbatim_on_skip:
                    gated_chunk_id = self.verbatim.store(
                        text=text,
                        conversation_id=conversation_id,
                        turn_number=turn_number,
                        source_type=source_type,
                        timestamp=timestamp,
                    )
                return IngestResult(
                    entities_created=[],
                    entities_resolved=[],
                    relationships_created=0,
                    conflicts_detected=0,
                    verbatim_chunk_id=gated_chunk_id,
                    gated_out=True,
                    gate_reason=decision.reason,
                )

        source_ref = SourceRef(
            conversation_id=conversation_id,
            turn_number=turn_number,
            verbatim=text,
            authority=authority,
        )
        if timestamp:
            source_ref.timestamp = timestamp

        # 1. Store verbatim text
        chunk_id = self.verbatim.store(
            text=text,
            conversation_id=conversation_id,
            turn_number=turn_number,
            source_type=source_type,
            timestamp=timestamp,
        )

        # 2. Extract entities
        extracted_entities = self._entity_extractor.extract(text)

        # 3. Resolve entities against graph
        created = []
        resolved = []
        entity_map: dict[str, Entity] = {}

        for ee in extracted_entities:
            resolution = self._resolver.resolve(ee)

            if resolution.action == "merge" and resolution.entity:
                entity = resolution.entity
                resolved.append(entity.name)
                if resolution.add_alias:
                    self._db.execute(
                        "INSERT INTO entity_aliases (id, entity_id, alias, added_at) VALUES (?, ?, ?, ?)",
                        (_new_id(), entity.id, resolution.add_alias, _now()),
                    )
                    self._db.conn.commit()
            else:
                entity = self.graph.create_entity(
                    ee.name, ee.type, source_ref=source_ref
                )
                created.append(entity.name)

            entity_map[ee.name] = entity

            # Set properties with conflict detection
            prop_as_of = timestamp  # None falls back to _now() in set_property
            for key, value in ee.properties.items():
                conflict_result = self._conflict_detector.check(
                    entity.id, key, value, new_authority=authority
                )
                if conflict_result.conflict_type in (
                    ConflictType.CONFIRMATION,
                    ConflictType.UPDATE,
                ):
                    self.graph.set_property(
                        entity.id, key, value,
                        as_of=prop_as_of, source_ref=source_ref,
                    )
                elif conflict_result.conflict_type == ConflictType.CONTRADICTION:
                    # Store the new value but it's flagged
                    new_prop = self.graph.set_property(
                        entity.id, key, value,
                        as_of=prop_as_of, source_ref=source_ref,
                    )
                    # Update the conflict record with the new property's ID
                    if conflict_result.conflict_id and new_prop:
                        self._db.execute(
                            "UPDATE conflicts SET value_b_id = ? WHERE id = ?",
                            (new_prop.id, conflict_result.conflict_id),
                        )
                        self._db.conn.commit()
                elif conflict_result.conflict_type == ConflictType.HISTORICAL:
                    # Store as historical
                    self.graph.set_property(
                        entity.id, key, value,
                        as_of=source_ref.timestamp,
                        source_ref=source_ref,
                    )

        # 4. Extract and create relationships
        extracted_relations = self._relation_extractor.extract(text, extracted_entities)
        rels_created = 0
        for er in extracted_relations:
            source_entity = entity_map.get(er.source)
            target_entity = entity_map.get(er.target)
            if source_entity and target_entity:
                existing = self.graph.find_relationship(
                    source_entity.id, target_entity.id, er.type
                )
                if not existing:
                    self.graph.create_relationship(
                        source_entity.id,
                        target_entity.id,
                        er.type,
                        source_ref=source_ref,
                    )
                    rels_created += 1

        conflicts = self._conflict_detector.get_unresolved()

        # Post-ingestion consolidation
        self._ingestion_count += 1
        if self._ingestion_count % self.config.consolidation.decay_interval_ingestions == 0:
            self._consolidation.run_quick()
        if self._ingestion_count % self.config.consolidation.full_interval_ingestions == 0:
            self._consolidation.run_full()

        return IngestResult(
            entities_created=created,
            entities_resolved=resolved,
            relationships_created=rels_created,
            conflicts_detected=len(conflicts),
            verbatim_chunk_id=chunk_id,
        )

    # ── Retrieval ────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        context: str = "",
        token_budget: int | None = None,
        as_of: str | None = None,
    ) -> MemoryContext:
        """Retrieve relevant memories for a query."""
        return self._retrieval.recall(query, context, token_budget, as_of)

    def recall_entity(self, entity_id: str, depth: int = 2) -> Entity | None:
        """Get everything known about a specific entity.

        When depth > 0, also populates the entity's relationships and
        fetches neighboring entities up to *depth* hops away.
        """
        entity = self.graph.get_entity(entity_id)
        if entity and depth > 0:
            # Attach neighbors up to the requested hop depth
            entity._neighbors = self.graph.get_neighbors(entity_id, max_hops=depth)
        return entity

    # ── Direct Manipulation ──────────────────────────────────────

    def correct(
        self, entity_id: str, property_key: str, new_value: object, reason: str = ""
    ) -> None:
        """Explicitly correct a fact (user override)."""
        from memento.models import StatementType

        source_ref = SourceRef(
            statement_type=StatementType.CORRECTED,
            verbatim=f"User correction: {property_key} = {new_value}. {reason}",
            authority=1.0,
        )
        self.graph.set_property(
            entity_id, property_key, new_value, source_ref=source_ref
        )

    def forget(self, entity_id: str | None = None, relationship_id: str | None = None) -> None:
        """Remove a memory (soft delete — archives the entity or ends the relationship)."""
        if entity_id:
            self._db.execute(
                "UPDATE entities SET archived = 1 WHERE id = ?", (entity_id,)
            )
            self._db.conn.commit()
        if relationship_id:
            self._db.execute(
                "UPDATE relationships SET valid_to = ? WHERE id = ?",
                (_now(), relationship_id),
            )
            self._db.conn.commit()

    def merge(self, entity_a_id: str, entity_b_id: str) -> dict:
        """Merge two entities."""
        return self.graph.merge_entities(entity_a_id, entity_b_id)

    # ── Introspection ────────────────────────────────────────────

    def conflicts(self) -> list[Conflict]:
        """List all unresolved contradictions."""
        return self._conflict_detector.get_unresolved()

    def entity_list(self, type: EntityType | None = None) -> list[Entity]:
        """List known entities, optionally filtered by type."""
        return self.graph.search_entities(type=type)

    def health(self) -> GraphHealth:
        """Get graph health statistics."""
        stats = self.graph.stats()
        unresolved = len(self._conflict_detector.get_unresolved())
        return GraphHealth(
            node_count=stats["node_count"],
            edge_count=stats["edge_count"],
            active_property_count=stats["active_property_count"],
            avg_confidence=stats["avg_confidence"],
            unresolved_conflicts=unresolved,
        )

    # ── Privacy & Audit ────────────────────────────────────────────

    def export_entity_data(self, entity_id: str) -> EntityDataExport | None:
        """Full data export for an entity (GDPR data subject access)."""
        return self._privacy.export_entity_data(entity_id)

    def audit_belief(self, entity_id: str, property_key: str) -> BeliefChain:
        """Trace a belief back through its full provenance chain."""
        return self._privacy.audit_belief(entity_id, property_key)

    def hard_delete(self, entity_id: str) -> DeletionReceipt | None:
        """True hard delete with compliance receipt. Irreversible."""
        return self._privacy.delete_entity_cascade(entity_id)

    # ── Consolidation ────────────────────────────────────────────

    def consolidate(self) -> ConsolidationResult:
        """Run a full consolidation pass (decay, merge, prune, index)."""
        return self._consolidation.run_full()

    # ── Session Management ───────────────────────────────────────

    def start_session(self) -> Session:
        """Start a new conversation session with scratchpad tracking."""
        return Session(self)
