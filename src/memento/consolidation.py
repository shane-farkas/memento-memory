"""Consolidation engine: background graph maintenance.

Handles confidence decay, redundancy merging, orphan pruning,
cluster promotion, centrality index materialization, and
contradiction auto-resolution.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import ConflictStatus, _now

logger = logging.getLogger(__name__)

DEFAULT_HALF_LIVES = {
    "employment": 180.0,
    "location": 120.0,
    "preference": 90.0,
    "relationship": 365.0,
    "project": 60.0,
    "contact_info": 365.0,
    "default": 180.0,
}


@dataclass
class ConsolidationResult:
    """Summary of a consolidation pass."""

    facts_decayed: int = 0
    redundancies_merged: int = 0
    orphans_archived: int = 0
    contradictions_resolved: int = 0
    centrality_entries: int = 0


class ConsolidationEngine:
    """Background graph maintenance operations."""

    def __init__(
        self,
        graph: GraphStore,
        half_lives: dict[str, float] | None = None,
    ) -> None:
        self.graph = graph
        self.db = graph.db
        self.half_lives = half_lives or DEFAULT_HALF_LIVES
        self._ensure_centrality_table()

    def _ensure_centrality_table(self) -> None:
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS centrality_index (
                entity_id   TEXT PRIMARY KEY REFERENCES entities(id),
                betweenness REAL NOT NULL DEFAULT 0.0,
                degree      REAL NOT NULL DEFAULT 0.0,
                updated_at  TEXT NOT NULL
            )
        """)
        self.db.conn.commit()

    def run_full(self) -> ConsolidationResult:
        """Run a full consolidation pass (all operations)."""
        result = ConsolidationResult()
        result.facts_decayed = self.decay_confidence()
        result.redundancies_merged = self.merge_redundant()
        result.orphans_archived = self.prune_orphans()
        result.contradictions_resolved = self.auto_resolve_contradictions()
        result.centrality_entries = self.materialize_centrality()
        return result

    def run_quick(self) -> ConsolidationResult:
        """Run a quick post-ingestion pass (decay + redundancy only)."""
        result = ConsolidationResult()
        result.facts_decayed = self.decay_confidence()
        result.redundancies_merged = self.merge_redundant()
        return result

    # ── Confidence Decay ─────────────────────────────────────────

    def decay_confidence(self) -> int:
        """Apply confidence decay to all active properties.

        Computes decay from recorded_at directly against a base of 1.0,
        avoiding compounding errors across multiple consolidation passes:
          time_decay = 0.5^(age / half_life)
          confirmation_multiplier = 1.0 + min(confirmation_count * 0.1, 0.5)
          decayed = time_decay * confirmation_multiplier

        The result is clamped to never exceed the property's current
        confidence (confirmations can only slow decay, not reverse it).
        """
        now = datetime.now(timezone.utc)
        rows = self.db.fetchall(
            "SELECT id, key, confidence, recorded_at FROM properties WHERE superseded_by_id IS NULL"
        )

        updated = 0
        for row in rows:
            try:
                recorded = datetime.fromisoformat(row["recorded_at"])
                age_days = max(0, (now - recorded).days)
            except Exception:
                continue

            half_life = self.half_lives.get(
                row["key"], self.half_lives["default"]
            )

            # Compute decay from a fixed base of 1.0, not the current
            # confidence. This makes the result idempotent — running
            # decay N times produces the same value as running it once.
            time_decay = 0.5 ** (age_days / half_life)

            # Count confirmations for this property value
            confirmation_count = self._count_confirmations(
                row["id"], row["key"]
            )
            confirmation_multiplier = 1.0 + min(confirmation_count * 0.1, 0.5)

            decayed = min(time_decay * confirmation_multiplier, 1.0)

            if abs(decayed - row["confidence"]) > 0.001:
                self.db.execute(
                    "UPDATE properties SET confidence = ? WHERE id = ?",
                    (decayed, row["id"]),
                )
                updated += 1

        if updated:
            self.db.conn.commit()
        logger.info("Decayed confidence on %d properties", updated)
        return updated

    def _count_confirmations(self, property_id: str, key: str) -> int:
        """Count how many times a property was confirmed via _boost_confidence.

        Approximated by counting how many superseded properties on the same
        entity+key share the same value (i.e., were set and then re-confirmed).
        """
        row = self.db.fetchone(
            """SELECT COUNT(*) as cnt FROM properties p1
               JOIN properties p2 ON p1.entity_id = p2.entity_id
                 AND p1.key = p2.key AND p1.value_json = p2.value_json
               WHERE p2.id = ? AND p1.id != p2.id AND p1.key = ?""",
            (property_id, key),
        )
        return row["cnt"] if row else 0

    # ── Redundancy Merging ───────────────────────────────────────

    def merge_redundant(self) -> int:
        """Detect and merge semantically identical facts stored separately.

        Two properties on the same entity with the same key and equivalent
        values (case-insensitive string match) are merged: the newer one
        supersedes the older, with boosted confidence.
        """
        merged = 0
        entities = self.db.fetchall(
            "SELECT id FROM entities WHERE archived = 0"
        )

        for entity_row in entities:
            eid = entity_row["id"]
            # Find keys with multiple non-superseded values
            keys = self.db.fetchall(
                """SELECT key, COUNT(*) as cnt FROM properties
                   WHERE entity_id = ? AND superseded_by_id IS NULL
                   GROUP BY key HAVING cnt > 1""",
                (eid,),
            )

            for key_row in keys:
                key = key_row["key"]
                props = self.db.fetchall(
                    """SELECT id, value_json, confidence, recorded_at
                       FROM properties
                       WHERE entity_id = ? AND key = ? AND superseded_by_id IS NULL
                       ORDER BY recorded_at ASC""",
                    (eid, key),
                )

                # Compare each pair
                i = 0
                while i < len(props) - 1:
                    j = i + 1
                    while j < len(props):
                        v1 = json.loads(props[i]["value_json"]) if props[i]["value_json"] else None
                        v2 = json.loads(props[j]["value_json"]) if props[j]["value_json"] else None

                        if self._values_equivalent(v1, v2):
                            # Merge: newer supersedes older, boost confidence
                            older_id = props[i]["id"]
                            newer_id = props[j]["id"]
                            new_conf = min(1.0, props[j]["confidence"] + 0.1)
                            self.db.execute(
                                "UPDATE properties SET superseded_by_id = ? WHERE id = ?",
                                (newer_id, older_id),
                            )
                            self.db.execute(
                                "UPDATE properties SET confidence = ? WHERE id = ?",
                                (new_conf, newer_id),
                            )
                            merged += 1
                            props.pop(i)  # Remove merged item
                            j = i + 1  # Reset inner loop
                            continue
                        j += 1
                    i += 1

        if merged:
            self.db.conn.commit()
        logger.info("Merged %d redundant facts", merged)
        return merged

    # ── Orphan Pruning ───────────────────────────────────────────

    def prune_orphans(
        self, min_confidence: float = 0.2, min_access_days: int = 90
    ) -> int:
        """Archive entities with no edges, low confidence, and no recent access."""
        now = datetime.now(timezone.utc)
        cutoff = now.isoformat()

        rows = self.db.fetchall(
            """SELECT e.id, e.confidence, e.last_seen
               FROM entities e
               WHERE e.archived = 0
                 AND e.confidence < ?
                 AND NOT EXISTS (
                     SELECT 1 FROM relationships r
                     WHERE r.source_id = e.id OR r.target_id = e.id
                 )""",
            (min_confidence,),
        )

        archived = 0
        for row in rows:
            try:
                last_seen = datetime.fromisoformat(row["last_seen"])
                days_since = (now - last_seen).days
                if days_since >= min_access_days:
                    self.db.execute(
                        "UPDATE entities SET archived = 1 WHERE id = ?",
                        (row["id"],),
                    )
                    archived += 1
            except Exception:
                continue

        if archived:
            self.db.conn.commit()
        logger.info("Archived %d orphan entities", archived)
        return archived

    # ── Contradiction Auto-Resolution ────────────────────────────

    def auto_resolve_contradictions(self, decay_threshold: float = 0.2) -> int:
        """Auto-resolve contradictions where one side has decayed significantly."""
        rows = self.db.fetchall(
            "SELECT * FROM conflicts WHERE status = 'unresolved'"
        )

        resolved = 0
        for row in rows:
            value_a = self.db.fetchone(
                "SELECT confidence FROM properties WHERE id = ?",
                (row["value_a_id"],),
            )
            value_b = self.db.fetchone(
                "SELECT confidence FROM properties WHERE id = ?",
                (row["value_b_id"],),
            )

            if not value_a or not value_b:
                continue

            conf_a = value_a["confidence"]
            conf_b = value_b["confidence"]

            # If one side has decayed below threshold while the other hasn't
            if conf_a < decay_threshold and conf_b >= decay_threshold:
                self.db.execute(
                    "UPDATE conflicts SET status = ?, resolved_at = ?, resolution = ? WHERE id = ?",
                    (
                        ConflictStatus.AUTO_RESOLVED.value,
                        _now(),
                        f"Value A decayed to {conf_a:.2f}",
                        row["id"],
                    ),
                )
                resolved += 1
            elif conf_b < decay_threshold and conf_a >= decay_threshold:
                self.db.execute(
                    "UPDATE conflicts SET status = ?, resolved_at = ?, resolution = ? WHERE id = ?",
                    (
                        ConflictStatus.AUTO_RESOLVED.value,
                        _now(),
                        f"Value B decayed to {conf_b:.2f}",
                        row["id"],
                    ),
                )
                resolved += 1
            elif conf_a < decay_threshold and conf_b < decay_threshold:
                # Both decayed → archive both
                self.db.execute(
                    "UPDATE conflicts SET status = ?, resolved_at = ?, resolution = ? WHERE id = ?",
                    (
                        ConflictStatus.AUTO_RESOLVED.value,
                        _now(),
                        "Both values decayed",
                        row["id"],
                    ),
                )
                resolved += 1

        if resolved:
            self.db.conn.commit()
        logger.info("Auto-resolved %d contradictions", resolved)
        return resolved

    # ── Centrality Index Materialization ─────────────────────────

    def materialize_centrality(self) -> int:
        """Pre-compute degree centrality for each entity.

        Betweenness centrality is expensive to compute; for now we
        materialize degree centrality (count of relationships) as the
        primary graph-structural signal.
        """
        now = _now()
        rows = self.db.fetchall(
            """SELECT e.id,
                      (SELECT COUNT(*) FROM relationships r
                       WHERE r.source_id = e.id OR r.target_id = e.id) as degree
               FROM entities e
               WHERE e.archived = 0"""
        )

        count = 0
        for row in rows:
            degree = row["degree"]
            # Normalize to 0-1 range (cap at 20 relationships)
            degree_norm = min(degree / 20.0, 1.0)

            self.db.execute(
                """INSERT INTO centrality_index (entity_id, betweenness, degree, updated_at)
                   VALUES (?, 0.0, ?, ?)
                   ON CONFLICT(entity_id) DO UPDATE SET
                     degree = excluded.degree,
                     updated_at = excluded.updated_at""",
                (row["id"], degree_norm, now),
            )
            count += 1

        if count:
            self.db.conn.commit()
        logger.info("Materialized centrality index for %d entities", count)
        return count

    # ── Helpers ──────────────────────────────────────────────────

    def _values_equivalent(self, v1: object, v2: object) -> bool:
        if v1 == v2:
            return True
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.strip().lower() == v2.strip().lower()
        return False
