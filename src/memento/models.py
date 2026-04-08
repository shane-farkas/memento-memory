"""Core data models for the Memento knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from uuid_extensions import uuid7


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid7())


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"


class StatementType(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    CORRECTED = "corrected"
    THIRD_PARTY = "third_party"


class ConflictStatus(str, Enum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    AUTO_RESOLVED = "auto_resolved"


@dataclass
class SourceRef:
    """Provenance tracking for where a fact came from."""

    id: str = field(default_factory=_new_id)
    conversation_id: str = ""
    turn_number: int = 0
    timestamp: str = field(default_factory=_now)
    statement_type: StatementType = StatementType.EXPLICIT
    verbatim: str = ""
    authority: float = 1.0


@dataclass
class PropertyValue:
    """A single versioned property value with temporal and provenance metadata."""

    id: str = field(default_factory=_new_id)
    entity_id: str = ""
    key: str = ""
    value: Any = None
    as_of: str = field(default_factory=_now)
    recorded_at: str = field(default_factory=_now)
    source_ref_id: str = ""
    confidence: float = 1.0
    superseded_by_id: str | None = None


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str = field(default_factory=_new_id)
    type: EntityType = EntityType.PERSON
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    last_seen: str = field(default_factory=_now)
    access_count: int = 0
    confidence: float = 1.0
    archived: bool = False
    merged_into: str | None = None
    properties: dict[str, PropertyValue] = field(default_factory=dict)


@dataclass
class Relationship:
    """An edge in the knowledge graph."""

    id: str = field(default_factory=_new_id)
    source_id: str = ""
    target_id: str = ""
    type: str = ""
    valid_from: str = field(default_factory=_now)
    valid_to: str | None = None
    source_ref_id: str = ""
    confidence: float = 1.0
    access_count: int = 0
    properties: dict[str, PropertyValue] = field(default_factory=dict)


@dataclass
class Conflict:
    """An unresolved contradiction between two property values."""

    id: str = field(default_factory=_new_id)
    entity_id: str = ""
    property_key: str = ""
    value_a_id: str = ""
    value_b_id: str = ""
    status: ConflictStatus = ConflictStatus.UNRESOLVED
    created_at: str = field(default_factory=_now)
    resolved_at: str | None = None
    resolution: str | None = None


@dataclass
class MergeLog:
    """Audit record of an entity merge operation."""

    id: str = field(default_factory=_new_id)
    survivor_id: str = ""
    absorbed_id: str = ""
    timestamp: str = field(default_factory=_now)
    reason: str = ""
    undo_data: str = ""  # JSON blob for reversibility
