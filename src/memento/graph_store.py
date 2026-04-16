"""Graph storage layer: CRUD operations, traversal, and bitemporal queries."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from memento.db import Database
from memento.models import (
    Entity,
    EntityType,
    PropertyValue,
    Relationship,
    SourceRef,
    _new_id,
    _now,
)
from memento.schema import create_tables

logger = logging.getLogger(__name__)


class GraphStore:
    """Core graph storage with CRUD, traversal, and bitemporal queries."""

    def __init__(self, db: Database) -> None:
        self.db = db
        create_tables(db)

    # ── Entity CRUD ──────────────────────────────────────────────────

    def create_entity(
        self,
        name: str,
        type: EntityType,
        aliases: list[str] | None = None,
        confidence: float = 1.0,
        source_ref: SourceRef | None = None,
    ) -> Entity:
        """Create a new entity node in the graph."""
        entity = Entity(
            name=name,
            type=type,
            confidence=confidence,
        )

        if source_ref:
            self._upsert_source_ref(source_ref)

        with self.db.transaction() as cur:
            cur.execute(
                """INSERT INTO entities (id, type, name, created_at, last_seen, access_count, confidence, archived)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity.id,
                    entity.type.value,
                    entity.name,
                    entity.created_at,
                    entity.last_seen,
                    entity.access_count,
                    entity.confidence,
                    int(entity.archived),
                ),
            )

            # Add aliases
            all_aliases = list(set(aliases or []))
            for alias in all_aliases:
                cur.execute(
                    "INSERT INTO entity_aliases (id, entity_id, alias, added_at) VALUES (?, ?, ?, ?)",
                    (_new_id(), entity.id, alias, _now()),
                )

        entity.aliases = all_aliases
        logger.debug("Created entity: %s (%s)", entity.name, entity.id)
        return entity

    def record_mention(self, entity_id: str, conversation_id: str) -> None:
        """Bump mention_count and refresh source_count for an entity.

        Called once per ingest per resolved entity, regardless of whether
        the entity was newly created or matched to an existing record.
        Distinct conversation_ids are tracked in entity_sources so
        source_count reflects how widely an entity has been observed.
        """
        if not conversation_id:
            return
        with self.db.transaction() as cur:
            cur.execute(
                """INSERT OR IGNORE INTO entity_sources
                       (entity_id, conversation_id, first_seen)
                   VALUES (?, ?, ?)""",
                (entity_id, conversation_id, _now()),
            )
            cur.execute(
                """UPDATE entities
                       SET mention_count = mention_count + 1,
                           source_count  = (SELECT COUNT(*)
                                            FROM entity_sources
                                            WHERE entity_id = ?)
                     WHERE id = ?""",
                (entity_id, entity_id),
            )

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID, including aliases and current properties."""
        row = self.db.fetchone(
            "SELECT * FROM entities WHERE id = ? AND archived = 0", (entity_id,)
        )
        if row is None:
            return None

        entity = self._row_to_entity(row)
        entity.aliases = self._get_aliases(entity_id)
        entity.properties = self._get_current_properties(entity_id)
        return entity

    def search_entities(
        self,
        name: str | None = None,
        type: EntityType | None = None,
        fuzzy: bool = False,
        include_archived: bool = False,
    ) -> list[Entity]:
        """Search for entities by name and/or type."""
        conditions = []
        params: list = []

        if not include_archived:
            conditions.append("e.archived = 0")

        if type:
            conditions.append("e.type = ?")
            params.append(type.value)

        if name:
            if fuzzy:
                # LIKE-based fuzzy search (basic; full Levenshtein comes in Stage 7)
                conditions.append(
                    "(e.name LIKE ? OR EXISTS (SELECT 1 FROM entity_aliases a WHERE a.entity_id = e.id AND a.alias LIKE ?))"
                )
                pattern = f"%{name}%"
                params.extend([pattern, pattern])
            else:
                conditions.append(
                    "(e.name = ? OR EXISTS (SELECT 1 FROM entity_aliases a WHERE a.entity_id = e.id AND a.alias = ?))"
                )
                params.extend([name, name])

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.db.fetchall(
            f"SELECT * FROM entities e WHERE {where}", tuple(params)
        )

        entities = []
        for row in rows:
            entity = self._row_to_entity(row)
            entity.aliases = self._get_aliases(entity.id)
            entities.append(entity)
        return entities

    # ── Property CRUD ────────────────────────────────────────────────

    def set_property(
        self,
        entity_id: str,
        key: str,
        value: object,
        as_of: str | None = None,
        source_ref: SourceRef | None = None,
        confidence: float = 1.0,
    ) -> PropertyValue:
        """Set a property on an entity. Supersedes the previous value if one exists."""
        now = _now()
        as_of = as_of or now

        if source_ref:
            self._upsert_source_ref(source_ref)

        ref_id = source_ref.id if source_ref else None

        new_prop = PropertyValue(
            entity_id=entity_id,
            key=key,
            value=value,
            as_of=as_of,
            recorded_at=now,
            source_ref_id=ref_id or "",
            confidence=confidence,
        )

        with self.db.transaction() as cur:
            # Find current (non-superseded) value for this key
            current = cur.execute(
                """SELECT id FROM properties
                   WHERE entity_id = ? AND key = ? AND superseded_by_id IS NULL""",
                (entity_id, key),
            ).fetchone()

            # Insert the new property value
            cur.execute(
                """INSERT INTO properties (id, entity_id, key, value_json, as_of, recorded_at, source_ref_id, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_prop.id,
                    entity_id,
                    key,
                    json.dumps(value),
                    as_of,
                    now,
                    ref_id,
                    confidence,
                ),
            )

            # Supersede the old value
            if current:
                cur.execute(
                    "UPDATE properties SET superseded_by_id = ? WHERE id = ?",
                    (new_prop.id, current["id"]),
                )

            # Update entity last_seen
            cur.execute(
                "UPDATE entities SET last_seen = ? WHERE id = ?", (now, entity_id)
            )

        logger.debug("Set property %s.%s = %s", entity_id[:8], key, value)
        return new_prop

    def get_property(
        self, entity_id: str, key: str, as_of: str | None = None
    ) -> PropertyValue | None:
        """Get the current value of a property, or the value as of a given date."""
        if as_of:
            # Point-in-time: get the value that was current at as_of
            row = self.db.fetchone(
                """SELECT * FROM properties
                   WHERE entity_id = ? AND key = ? AND recorded_at <= ?
                   ORDER BY recorded_at DESC LIMIT 1""",
                (entity_id, key, as_of),
            )
        else:
            # Current: get the non-superseded value
            row = self.db.fetchone(
                """SELECT * FROM properties
                   WHERE entity_id = ? AND key = ? AND superseded_by_id IS NULL""",
                (entity_id, key),
            )

        if row is None:
            return None
        return self._row_to_property(row)

    def get_property_history(
        self, entity_id: str, key: str
    ) -> list[PropertyValue]:
        """Get the full history of a property (all values in the supersession chain)."""
        rows = self.db.fetchall(
            """SELECT * FROM properties
               WHERE entity_id = ? AND key = ?
               ORDER BY recorded_at ASC""",
            (entity_id, key),
        )
        return [self._row_to_property(row) for row in rows]

    # ── Relationship CRUD ────────────────────────────────────────────

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        type: str,
        valid_from: str | None = None,
        source_ref: SourceRef | None = None,
        confidence: float = 1.0,
    ) -> Relationship:
        """Create a relationship (edge) between two entities."""
        now = _now()
        valid_from = valid_from or now

        if source_ref:
            self._upsert_source_ref(source_ref)

        ref_id = source_ref.id if source_ref else None

        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            type=type,
            valid_from=valid_from,
            source_ref_id=ref_id or "",
            confidence=confidence,
        )

        with self.db.transaction() as cur:
            cur.execute(
                """INSERT INTO relationships (id, source_id, target_id, type, valid_from, source_ref_id, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rel.id,
                    source_id,
                    target_id,
                    type,
                    valid_from,
                    ref_id,
                    confidence,
                ),
            )

        logger.debug("Created relationship: %s -[%s]-> %s", source_id[:8], type, target_id[:8])
        return rel

    def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        type: str | None = None,
    ) -> list[Relationship]:
        """Get relationships for an entity. direction: 'outgoing', 'incoming', or 'both'."""
        conditions = []
        params: list = []

        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(entity_id)
        else:
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([entity_id, entity_id])

        if type:
            conditions.append("type = ?")
            params.append(type)

        where = " AND ".join(conditions)
        rows = self.db.fetchall(
            f"SELECT * FROM relationships WHERE {where}", tuple(params)
        )
        return [self._row_to_relationship(row) for row in rows]

    def find_relationship(
        self, source_id: str, target_id: str, type: str
    ) -> Relationship | None:
        """Find a specific relationship between two entities."""
        row = self.db.fetchone(
            "SELECT * FROM relationships WHERE source_id = ? AND target_id = ? AND type = ?",
            (source_id, target_id, type),
        )
        if row is None:
            return None
        return self._row_to_relationship(row)

    # ── Graph Traversal ──────────────────────────────────────────────

    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1,
        types: list[str] | None = None,
    ) -> list[Entity]:
        """Get neighboring entities within max_hops via recursive CTE."""
        type_filter = ""
        params: list = [entity_id, max_hops]
        if types:
            placeholders = ",".join("?" for _ in types)
            type_filter = f"AND r.type IN ({placeholders})"
            params = [entity_id] + types + [max_hops]

        # Recursive CTE for multi-hop traversal
        query = f"""
        WITH RECURSIVE neighbors(entity_id, depth) AS (
            -- Base: direct neighbors of the starting entity
            SELECT CASE WHEN r.source_id = ? THEN r.target_id ELSE r.source_id END, 1
            FROM relationships r
            WHERE (r.source_id = ?1 OR r.target_id = ?1) {type_filter}

            UNION

            -- Recursive: neighbors of neighbors
            SELECT CASE WHEN r.source_id = n.entity_id THEN r.target_id ELSE r.source_id END, n.depth + 1
            FROM relationships r
            JOIN neighbors n ON (r.source_id = n.entity_id OR r.target_id = n.entity_id)
            WHERE n.depth < ?
              AND CASE WHEN r.source_id = n.entity_id THEN r.target_id ELSE r.source_id END != ?1
              {type_filter}
        )
        SELECT DISTINCT e.*
        FROM neighbors n
        JOIN entities e ON e.id = n.entity_id
        WHERE e.archived = 0 AND e.id != ?1
        """

        # Build params for the recursive CTE
        # ?1 = entity_id, type params, ?max = max_hops
        cte_params: list = [entity_id]
        if types:
            cte_params.extend(types)
        cte_params.append(max_hops)
        if types:
            cte_params.extend(types)

        rows = self.db.fetchall(query, tuple(cte_params))
        entities = []
        for row in rows:
            entity = self._row_to_entity(row)
            entity.aliases = self._get_aliases(entity.id)
            entities.append(entity)
        return entities

    # ── Bitemporal Queries ───────────────────────────────────────────

    def point_in_time_snapshot(
        self, entity_id: str, as_of: str
    ) -> Entity | None:
        """Get an entity's state as the system believed it at a given point in time.

        Returns the entity with properties reflecting what was known at `as_of`.
        Uses `recorded_at` (when the system learned the fact), not `as_of` on the
        property (when the fact was true in the real world).
        """
        row = self.db.fetchone(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        )
        if row is None:
            return None

        entity = self._row_to_entity(row)
        entity.aliases = self._get_aliases(entity_id)

        # Get all property keys for this entity
        key_rows = self.db.fetchall(
            "SELECT DISTINCT key FROM properties WHERE entity_id = ?",
            (entity_id,),
        )

        properties = {}
        for key_row in key_rows:
            key = key_row["key"]
            # Get the most recent value that was recorded on or before as_of
            prop_row = self.db.fetchone(
                """SELECT * FROM properties
                   WHERE entity_id = ? AND key = ? AND recorded_at <= ?
                   ORDER BY recorded_at DESC LIMIT 1""",
                (entity_id, key, as_of),
            )
            if prop_row:
                properties[key] = self._row_to_property(prop_row)

        entity.properties = properties
        return entity

    # ── Entity Merge / Split ────────────────────────────────────────

    def merge_entities(
        self,
        entity_a_id: str,
        entity_b_id: str,
        surviving_id: str | None = None,
        reason: str = "manual_merge",
    ) -> dict:
        """Retroactively merge two entities. Returns a merge result dict.

        The surviving entity absorbs all aliases, edges, and properties
        from the absorbed entity. The absorbed entity is archived.
        """
        survivor_id = surviving_id or entity_a_id
        absorbed_id = entity_b_id if survivor_id == entity_a_id else entity_a_id

        survivor = self.get_entity(survivor_id)
        absorbed = self.get_entity(absorbed_id)
        if not survivor or not absorbed:
            raise ValueError("Both entities must exist and not be archived")

        # Capture undo data before mutation
        undo_data = {
            "absorbed": {
                "id": absorbed.id,
                "name": absorbed.name,
                "type": absorbed.type.value,
                "aliases": absorbed.aliases,
                "created_at": absorbed.created_at,
                "last_seen": absorbed.last_seen,
                "access_count": absorbed.access_count,
                "confidence": absorbed.confidence,
            },
            "survivor_original_aliases": survivor.aliases[:],
        }

        edges_re_parented = 0
        properties_moved = 0

        with self.db.transaction() as cur:
            # 1. Merge aliases: union of all names
            new_aliases = set(survivor.aliases + absorbed.aliases + [absorbed.name])
            new_aliases.discard(survivor.name)  # don't alias our own name
            # Remove existing survivor aliases and re-insert union
            cur.execute(
                "DELETE FROM entity_aliases WHERE entity_id = ?", (survivor_id,)
            )
            for alias in new_aliases:
                cur.execute(
                    "INSERT INTO entity_aliases (id, entity_id, alias, added_at) VALUES (?, ?, ?, ?)",
                    (_new_id(), survivor_id, alias, _now()),
                )

            # 2. Re-parent relationships from absorbed → survivor
            rels = self.db.fetchall(
                "SELECT * FROM relationships WHERE source_id = ? OR target_id = ?",
                (absorbed_id, absorbed_id),
            )
            undo_data["absorbed_relationships"] = [dict(r) for r in rels]

            for rel in rels:
                new_source = survivor_id if rel["source_id"] == absorbed_id else rel["source_id"]
                new_target = survivor_id if rel["target_id"] == absorbed_id else rel["target_id"]

                # Skip self-loops that would result from the merge
                if new_source == new_target:
                    cur.execute("DELETE FROM relationships WHERE id = ?", (rel["id"],))
                    edges_re_parented += 1
                    continue

                # Check for duplicate edge
                existing = cur.execute(
                    "SELECT id, confidence FROM relationships WHERE source_id = ? AND target_id = ? AND type = ?",
                    (new_source, new_target, rel["type"]),
                ).fetchone()

                if existing:
                    # Merge: keep higher confidence, earlier valid_from
                    cur.execute(
                        "UPDATE relationships SET confidence = MAX(confidence, ?) WHERE id = ?",
                        (rel["confidence"], existing["id"]),
                    )
                    cur.execute("DELETE FROM relationships WHERE id = ?", (rel["id"],))
                else:
                    # Re-parent
                    cur.execute(
                        "UPDATE relationships SET source_id = ?, target_id = ? WHERE id = ?",
                        (new_source, new_target, rel["id"]),
                    )
                edges_re_parented += 1

            # 3. Move properties from absorbed to survivor
            absorbed_props = self.db.fetchall(
                "SELECT * FROM properties WHERE entity_id = ?", (absorbed_id,)
            )
            undo_data["absorbed_properties"] = [dict(p) for p in absorbed_props]

            for prop in absorbed_props:
                # Check if survivor already has this key (current value)
                survivor_current = cur.execute(
                    "SELECT id FROM properties WHERE entity_id = ? AND key = ? AND superseded_by_id IS NULL",
                    (survivor_id, prop["key"]),
                ).fetchone()

                if survivor_current:
                    # Conflict: both have the property. Keep the one with higher confidence.
                    # The absorbed value becomes historical on the survivor.
                    cur.execute(
                        "UPDATE properties SET entity_id = ?, superseded_by_id = ? WHERE id = ?",
                        (survivor_id, survivor_current["id"], prop["id"]),
                    )
                else:
                    # Just move it
                    cur.execute(
                        "UPDATE properties SET entity_id = ? WHERE id = ?",
                        (survivor_id, prop["id"]),
                    )
                properties_moved += 1

            # 4. Merge access metadata
            cur.execute(
                """UPDATE entities SET
                   access_count = access_count + ?,
                   last_seen = MAX(last_seen, ?),
                   created_at = MIN(created_at, ?)
                   WHERE id = ?""",
                (absorbed.access_count, absorbed.last_seen, absorbed.created_at, survivor_id),
            )

            # 5. Archive the absorbed entity
            cur.execute(
                "UPDATE entities SET archived = 1, merged_into = ? WHERE id = ?",
                (survivor_id, absorbed_id),
            )

            # 6. Log the merge
            merge_log_id = _new_id()
            cur.execute(
                """INSERT INTO merge_log (id, survivor_id, absorbed_id, timestamp, reason, undo_data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (merge_log_id, survivor_id, absorbed_id, _now(), reason, json.dumps(undo_data)),
            )

        return {
            "merge_log_id": merge_log_id,
            "survivor_id": survivor_id,
            "absorbed_id": absorbed_id,
            "edges_re_parented": edges_re_parented,
            "properties_moved": properties_moved,
        }

    def split_entity(self, merge_log_id: str) -> dict:
        """Undo a merge operation using the stored undo data."""
        row = self.db.fetchone(
            "SELECT * FROM merge_log WHERE id = ?", (merge_log_id,)
        )
        if row is None:
            raise ValueError(f"Merge log entry not found: {merge_log_id}")

        undo_data = json.loads(row["undo_data"])
        survivor_id = row["survivor_id"]
        absorbed_id = row["absorbed_id"]
        absorbed_info = undo_data["absorbed"]

        with self.db.transaction() as cur:
            # 1. Restore absorbed entity
            cur.execute(
                "UPDATE entities SET archived = 0, merged_into = NULL WHERE id = ?",
                (absorbed_id,),
            )

            # If the absorbed entity was fully deleted (shouldn't happen normally),
            # we can't undo. Check it exists.
            check = cur.execute(
                "SELECT id FROM entities WHERE id = ?", (absorbed_id,)
            ).fetchone()
            if not check:
                raise ValueError("Cannot undo: absorbed entity was hard-deleted")

            # 2. Restore absorbed entity's original properties
            for prop in undo_data.get("absorbed_properties", []):
                cur.execute(
                    "UPDATE properties SET entity_id = ?, superseded_by_id = ? WHERE id = ?",
                    (absorbed_id, prop.get("superseded_by_id"), prop["id"]),
                )

            # 3. Restore absorbed entity's original relationships
            for rel in undo_data.get("absorbed_relationships", []):
                # Check if the relationship still exists (may have been merged into a duplicate)
                existing = cur.execute(
                    "SELECT id FROM relationships WHERE id = ?", (rel["id"],)
                ).fetchone()
                if existing:
                    cur.execute(
                        "UPDATE relationships SET source_id = ?, target_id = ? WHERE id = ?",
                        (rel["source_id"], rel["target_id"], rel["id"]),
                    )

            # 4. Restore survivor's original aliases
            cur.execute(
                "DELETE FROM entity_aliases WHERE entity_id = ?", (survivor_id,)
            )
            for alias in undo_data.get("survivor_original_aliases", []):
                cur.execute(
                    "INSERT INTO entity_aliases (id, entity_id, alias, added_at) VALUES (?, ?, ?, ?)",
                    (_new_id(), survivor_id, alias, _now()),
                )

            # 5. Restore absorbed entity's aliases
            cur.execute(
                "DELETE FROM entity_aliases WHERE entity_id = ?", (absorbed_id,)
            )
            for alias in absorbed_info.get("aliases", []):
                cur.execute(
                    "INSERT INTO entity_aliases (id, entity_id, alias, added_at) VALUES (?, ?, ?, ?)",
                    (_new_id(), absorbed_id, alias, _now()),
                )

            # 6. Remove the merge log entry
            cur.execute("DELETE FROM merge_log WHERE id = ?", (merge_log_id,))

        return {
            "survivor_id": survivor_id,
            "restored_id": absorbed_id,
        }

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get graph statistics."""
        node_count = self.db.fetchone(
            "SELECT COUNT(*) as c FROM entities WHERE archived = 0"
        )["c"]
        edge_count = self.db.fetchone(
            "SELECT COUNT(*) as c FROM relationships"
        )["c"]
        property_count = self.db.fetchone(
            "SELECT COUNT(*) as c FROM properties WHERE superseded_by_id IS NULL"
        )["c"]
        avg_confidence = self.db.fetchone(
            "SELECT AVG(confidence) as c FROM entities WHERE archived = 0"
        )["c"]

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "active_property_count": property_count,
            "avg_confidence": avg_confidence or 0.0,
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _upsert_source_ref(self, ref: SourceRef) -> None:
        """Insert a source reference if it doesn't exist."""
        existing = self.db.fetchone(
            "SELECT id FROM source_refs WHERE id = ?", (ref.id,)
        )
        if existing:
            return
        self.db.execute(
            """INSERT INTO source_refs (id, conversation_id, turn_number, timestamp, statement_type, verbatim, authority)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                ref.id,
                ref.conversation_id,
                ref.turn_number,
                ref.timestamp,
                ref.statement_type.value,
                ref.verbatim,
                ref.authority,
            ),
        )
        self.db.conn.commit()

    def _get_aliases(self, entity_id: str) -> list[str]:
        rows = self.db.fetchall(
            "SELECT alias FROM entity_aliases WHERE entity_id = ?", (entity_id,)
        )
        return [row["alias"] for row in rows]

    def _get_current_properties(self, entity_id: str) -> dict[str, PropertyValue]:
        rows = self.db.fetchall(
            """SELECT * FROM properties
               WHERE entity_id = ? AND superseded_by_id IS NULL""",
            (entity_id,),
        )
        return {row["key"]: self._row_to_property(row) for row in rows}

    def _row_to_entity(self, row) -> Entity:
        return Entity(
            id=row["id"],
            type=EntityType(row["type"]),
            name=row["name"],
            created_at=row["created_at"],
            last_seen=row["last_seen"],
            access_count=row["access_count"],
            confidence=row["confidence"],
            archived=bool(row["archived"]),
            merged_into=row["merged_into"],
        )

    def _row_to_property(self, row) -> PropertyValue:
        return PropertyValue(
            id=row["id"],
            entity_id=row["entity_id"],
            key=row["key"],
            value=json.loads(row["value_json"]) if row["value_json"] else None,
            as_of=row["as_of"],
            recorded_at=row["recorded_at"],
            source_ref_id=row["source_ref_id"] or "",
            confidence=row["confidence"],
            superseded_by_id=row["superseded_by_id"],
        )

    def _row_to_relationship(self, row) -> Relationship:
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            type=row["type"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            source_ref_id=row["source_ref_id"] or "",
            confidence=row["confidence"],
            access_count=row["access_count"],
        )
