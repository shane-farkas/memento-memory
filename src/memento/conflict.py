"""Conflict detection: identify when new facts contradict existing ones."""

from __future__ import annotations

import json
import logging
from enum import Enum

from memento.db import Database
from memento.graph_store import GraphStore
from memento.models import Conflict, ConflictStatus, PropertyValue, _new_id, _now

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    CONFIRMATION = "confirmation"
    UPDATE = "update"
    CONTRADICTION = "contradiction"
    HISTORICAL = "historical"
    RETRACTION = "retraction"


class ConflictResult:
    """Result of checking a new fact against existing facts."""

    def __init__(
        self,
        conflict_type: ConflictType,
        existing_value: PropertyValue | None = None,
        conflict_id: str | None = None,
    ) -> None:
        self.conflict_type = conflict_type
        self.existing_value = existing_value
        self.conflict_id = conflict_id


class ConflictDetector:
    """Detects conflicts when new facts are ingested.

    For each new fact (entity, property, value, timestamp), queries the graph
    for existing facts about the same entity+property and classifies the result.
    """

    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph

    def check(
        self,
        entity_id: str,
        key: str,
        new_value: object,
        new_as_of: str | None = None,
        new_authority: float = 1.0,
    ) -> ConflictResult:
        """Check a new fact against existing facts for the same entity+property.

        Returns a ConflictResult indicating what kind of interaction this is.
        """
        current = self.graph.get_property(entity_id, key)

        if current is None:
            # No existing value → this is a new fact, no conflict
            return ConflictResult(ConflictType.CONFIRMATION)

        new_as_of = new_as_of or _now()

        # Compare values
        if self._values_match(current.value, new_value):
            # Same value → confirmation, boost confidence
            self._boost_confidence(current)
            return ConflictResult(ConflictType.CONFIRMATION, existing_value=current)

        # Values differ — classify the conflict
        if new_as_of < current.as_of:
            # New fact has an older timestamp → it's historical context
            return ConflictResult(ConflictType.HISTORICAL, existing_value=current)

        if new_authority >= 0.8 and new_as_of >= current.as_of:
            # Newer + high authority → this is an update
            return ConflictResult(ConflictType.UPDATE, existing_value=current)

        # Ambiguous → record as a contradiction
        conflict_id = self._record_conflict(entity_id, key, current)
        return ConflictResult(
            ConflictType.CONTRADICTION,
            existing_value=current,
            conflict_id=conflict_id,
        )

    def get_unresolved(self, entity_id: str | None = None) -> list[Conflict]:
        """Get all unresolved conflicts, optionally filtered by entity."""
        if entity_id:
            rows = self.graph.db.fetchall(
                "SELECT * FROM conflicts WHERE entity_id = ? AND status = 'unresolved'",
                (entity_id,),
            )
        else:
            rows = self.graph.db.fetchall(
                "SELECT * FROM conflicts WHERE status = 'unresolved'"
            )
        return [self._row_to_conflict(row) for row in rows]

    def resolve(self, conflict_id: str, resolution: str) -> None:
        """Mark a conflict as resolved."""
        self.graph.db.execute(
            "UPDATE conflicts SET status = ?, resolved_at = ?, resolution = ? WHERE id = ?",
            (ConflictStatus.RESOLVED.value, _now(), resolution, conflict_id),
        )
        self.graph.db.conn.commit()

    def _values_match(self, v1: object, v2: object) -> bool:
        """Check if two values are semantically equivalent."""
        if v1 == v2:
            return True
        # Normalize strings for comparison
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.strip().lower() == v2.strip().lower()
        return False

    def _boost_confidence(self, prop: PropertyValue) -> None:
        """Boost confidence on a confirmed fact."""
        new_conf = min(1.0, prop.confidence + 0.05)
        self.graph.db.execute(
            "UPDATE properties SET confidence = ? WHERE id = ?",
            (new_conf, prop.id),
        )
        self.graph.db.conn.commit()

    def _record_conflict(
        self, entity_id: str, key: str, current: PropertyValue
    ) -> str:
        """Record an unresolved contradiction in the conflicts table."""
        conflict_id = _new_id()
        # The new value will be inserted separately; we record that the
        # current value is now in conflict. The value_b_id will be set
        # to the current property ID as a placeholder — the caller should
        # update it after inserting the new property value.
        self.graph.db.execute(
            """INSERT INTO conflicts (id, entity_id, property_key, value_a_id, value_b_id, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                conflict_id,
                entity_id,
                key,
                current.id,
                current.id,  # placeholder — updated after new value inserted
                ConflictStatus.UNRESOLVED.value,
                _now(),
            ),
        )
        self.graph.db.conn.commit()
        logger.info(
            "Recorded conflict for entity %s, property %s (conflict %s)",
            entity_id[:8],
            key,
            conflict_id[:8],
        )
        return conflict_id

    def _row_to_conflict(self, row) -> Conflict:
        return Conflict(
            id=row["id"],
            entity_id=row["entity_id"],
            property_key=row["property_key"],
            value_a_id=row["value_a_id"],
            value_b_id=row["value_b_id"],
            status=ConflictStatus(row["status"]),
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
            resolution=row["resolution"],
        )
