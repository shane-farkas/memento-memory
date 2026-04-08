"""Privacy and audit layer: GDPR-style data rights.

Provides entity data export, belief provenance tracing,
hard delete with compliance receipts, and access logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import _new_id, _now

logger = logging.getLogger(__name__)

ACCESS_LOG_DDL = """
CREATE TABLE IF NOT EXISTS access_log (
    id          TEXT PRIMARY KEY,
    entity_id   TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    query       TEXT NOT NULL DEFAULT '',
    caller      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_access_log_entity ON access_log(entity_id);
"""


@dataclass
class EntityDataExport:
    """Complete data export for an entity."""

    entity_id: str
    name: str
    type: str
    aliases: list[str]
    properties: list[dict]
    relationships: list[dict]
    source_refs: list[dict]
    access_log: list[dict]
    exported_at: str = field(default_factory=_now)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, default=str)


@dataclass
class BeliefChain:
    """Provenance chain tracing a belief back to its source."""

    entity_id: str
    property_key: str
    chain: list[dict]  # Ordered from current → historical


@dataclass
class DeletionReceipt:
    """Compliance receipt for a hard delete operation."""

    entity_id: str
    entity_name: str
    deleted_at: str
    items_deleted: dict[str, int]
    content_hash: str  # SHA-256 of deleted data for audit


class PrivacyLayer:
    """Handles data export, audit, and hard delete operations."""

    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        self.db = graph.db
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        for statement in ACCESS_LOG_DDL.split(";"):
            statement = statement.strip()
            if statement:
                self.db.execute(statement)
        self.db.conn.commit()

    # ── Access Logging ───────────────────────────────────────────

    def log_access(self, entity_id: str, query: str = "", caller: str = "") -> None:
        """Record an access event for an entity."""
        self.db.execute(
            "INSERT INTO access_log (id, entity_id, timestamp, query, caller) VALUES (?, ?, ?, ?, ?)",
            (_new_id(), entity_id, _now(), query, caller),
        )
        self.db.conn.commit()

    # ── Data Export ──────────────────────────────────────────────

    def export_entity_data(self, entity_id: str) -> EntityDataExport | None:
        """Export all data associated with an entity (GDPR data subject access)."""
        entity_row = self.db.fetchone(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        )
        if not entity_row:
            return None

        # Aliases
        alias_rows = self.db.fetchall(
            "SELECT alias, added_at FROM entity_aliases WHERE entity_id = ?",
            (entity_id,),
        )
        aliases = [r["alias"] for r in alias_rows]

        # Properties (all, including superseded)
        prop_rows = self.db.fetchall(
            "SELECT * FROM properties WHERE entity_id = ? ORDER BY recorded_at",
            (entity_id,),
        )
        properties = [dict(r) for r in prop_rows]

        # Relationships
        rel_rows = self.db.fetchall(
            "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
            (entity_id, entity_id),
        )
        relationships = [dict(r) for r in rel_rows]

        # Source refs linked to this entity's properties
        source_ref_ids = set()
        for p in prop_rows:
            if p["source_ref_id"]:
                source_ref_ids.add(p["source_ref_id"])
        for r in rel_rows:
            if r["source_ref_id"]:
                source_ref_ids.add(r["source_ref_id"])

        source_refs = []
        for ref_id in source_ref_ids:
            ref_row = self.db.fetchone(
                "SELECT * FROM source_refs WHERE id = ?", (ref_id,)
            )
            if ref_row:
                source_refs.append(dict(ref_row))

        # Access log
        access_rows = self.db.fetchall(
            "SELECT * FROM access_log WHERE entity_id = ? ORDER BY timestamp",
            (entity_id,),
        )
        access_log = [dict(r) for r in access_rows]

        return EntityDataExport(
            entity_id=entity_id,
            name=entity_row["name"],
            type=entity_row["type"],
            aliases=aliases,
            properties=properties,
            relationships=relationships,
            source_refs=source_refs,
            access_log=access_log,
        )

    # ── Belief Audit ─────────────────────────────────────────────

    def audit_belief(self, entity_id: str, property_key: str) -> BeliefChain:
        """Trace a belief back through its full provenance chain."""
        rows = self.db.fetchall(
            """SELECT p.*, sr.conversation_id, sr.turn_number, sr.verbatim,
                      sr.statement_type, sr.authority
               FROM properties p
               LEFT JOIN source_refs sr ON p.source_ref_id = sr.id
               WHERE p.entity_id = ? AND p.key = ?
               ORDER BY p.recorded_at DESC""",
            (entity_id, property_key),
        )

        chain = []
        for row in rows:
            entry = {
                "property_id": row["id"],
                "value": json.loads(row["value_json"]) if row["value_json"] else None,
                "as_of": row["as_of"],
                "recorded_at": row["recorded_at"],
                "confidence": row["confidence"],
                "superseded_by": row["superseded_by_id"],
                "source": {
                    "conversation_id": row["conversation_id"],
                    "turn_number": row["turn_number"],
                    "verbatim": row["verbatim"],
                    "statement_type": row["statement_type"],
                    "authority": row["authority"],
                } if row["conversation_id"] else None,
            }
            chain.append(entry)

        return BeliefChain(
            entity_id=entity_id,
            property_key=property_key,
            chain=chain,
        )

    # ── Hard Delete ──────────────────────────────────────────────

    def delete_entity_cascade(self, entity_id: str) -> DeletionReceipt | None:
        """True hard delete: remove all traces of an entity.

        Returns a DeletionReceipt with a cryptographic hash of deleted data
        for compliance auditing.
        """
        # First, export the data (for hashing)
        export = self.export_entity_data(entity_id)
        if not export:
            return None

        entity_name = export.name
        content_hash = hashlib.sha256(export.to_json().encode()).hexdigest()

        counts = {
            "properties": 0,
            "relationships": 0,
            "aliases": 0,
            "source_refs": 0,
            "access_log": 0,
            "verbatim_chunks": 0,
            "conflicts": 0,
            "centrality": 0,
        }

        with self.db.transaction() as cur:
            # Delete conflicts referencing this entity
            cur.execute(
                "DELETE FROM conflicts WHERE entity_id = ?", (entity_id,)
            )
            counts["conflicts"] = cur.rowcount

            # Delete centrality index (table may not exist if consolidation hasn't run)
            try:
                cur.execute(
                    "DELETE FROM centrality_index WHERE entity_id = ?", (entity_id,)
                )
                counts["centrality"] = cur.rowcount
            except Exception:
                pass

            # Collect source_ref IDs unique to this entity before deleting properties
            unique_refs = cur.execute(
                """SELECT DISTINCT p.source_ref_id FROM properties p
                   WHERE p.entity_id = ? AND p.source_ref_id IS NOT NULL
                   AND p.source_ref_id NOT IN (
                       SELECT source_ref_id FROM properties
                       WHERE entity_id != ? AND source_ref_id IS NOT NULL
                   )""",
                (entity_id, entity_id),
            ).fetchall()
            unique_ref_ids = [r[0] for r in unique_refs]

            # Delete properties
            cur.execute(
                "DELETE FROM properties WHERE entity_id = ?", (entity_id,)
            )
            counts["properties"] = cur.rowcount

            # Delete relationships
            cur.execute(
                "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )
            counts["relationships"] = cur.rowcount

            # Delete aliases
            cur.execute(
                "DELETE FROM entity_aliases WHERE entity_id = ?", (entity_id,)
            )
            counts["aliases"] = cur.rowcount

            # Delete unique source refs
            for ref_id in unique_ref_ids:
                cur.execute("DELETE FROM source_refs WHERE id = ?", (ref_id,))
                counts["source_refs"] += 1

            # Delete access log
            cur.execute(
                "DELETE FROM access_log WHERE entity_id = ?", (entity_id,)
            )
            counts["access_log"] = cur.rowcount

            # Delete the entity itself
            cur.execute("DELETE FROM entities WHERE id = ?", (entity_id,))

            # Delete merge log entries
            cur.execute(
                "DELETE FROM merge_log WHERE survivor_id = ? OR absorbed_id = ?",
                (entity_id, entity_id),
            )

        return DeletionReceipt(
            entity_id=entity_id,
            entity_name=entity_name,
            deleted_at=_now(),
            items_deleted=counts,
            content_hash=content_hash,
        )
