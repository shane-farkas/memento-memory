"""SQLite schema definition and migration for Memento."""

from __future__ import annotations

from memento.db import Database

SCHEMA_VERSION = 2

DDL = """
-- Core entity table
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL CHECK(type IN ('person','organization','project','location','concept','event')),
    name            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    confidence      REAL NOT NULL DEFAULT 1.0,
    archived        INTEGER NOT NULL DEFAULT 0,
    merged_into     TEXT REFERENCES entities(id),
    mention_count   INTEGER NOT NULL DEFAULT 0,
    source_count    INTEGER NOT NULL DEFAULT 0,
    tier            INTEGER NOT NULL DEFAULT 3
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_archived ON entities(archived);
CREATE INDEX IF NOT EXISTS idx_entities_tier ON entities(tier);

-- Distinct conversation sources that mentioned each entity.
-- Powers source_count and the future tier-promotion pass.
CREATE TABLE IF NOT EXISTS entity_sources (
    entity_id       TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    first_seen      TEXT NOT NULL,
    PRIMARY KEY (entity_id, conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_sources_conv ON entity_sources(conversation_id);

-- Entity aliases (many-to-one)
CREATE TABLE IF NOT EXISTS entity_aliases (
    id          TEXT PRIMARY KEY,
    entity_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    added_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_aliases_entity ON entity_aliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON entity_aliases(alias);

-- Source references (provenance tracking)
CREATE TABLE IF NOT EXISTS source_refs (
    id                  TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL DEFAULT '',
    turn_number         INTEGER NOT NULL DEFAULT 0,
    timestamp           TEXT NOT NULL,
    statement_type      TEXT NOT NULL DEFAULT 'explicit'
                        CHECK(statement_type IN ('explicit','inferred','corrected','third_party')),
    verbatim            TEXT NOT NULL DEFAULT '',
    authority           REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_source_refs_conversation ON source_refs(conversation_id);

-- Property values (versioned, with supersession chain)
CREATE TABLE IF NOT EXISTS properties (
    id                  TEXT PRIMARY KEY,
    entity_id           TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    key                 TEXT NOT NULL,
    value_json          TEXT,          -- JSON-encoded value
    as_of               TEXT NOT NULL, -- when the fact was true in the real world
    recorded_at         TEXT NOT NULL, -- when the system learned this fact
    source_ref_id       TEXT REFERENCES source_refs(id),
    confidence          REAL NOT NULL DEFAULT 1.0,
    superseded_by_id    TEXT REFERENCES properties(id)
);

CREATE INDEX IF NOT EXISTS idx_properties_entity_key ON properties(entity_id, key);
CREATE INDEX IF NOT EXISTS idx_properties_superseded ON properties(superseded_by_id);

-- Relationships (edges in the graph)
CREATE TABLE IF NOT EXISTS relationships (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id       TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type            TEXT NOT NULL,
    valid_from      TEXT NOT NULL,
    valid_to        TEXT,             -- NULL = still active
    source_ref_id   TEXT REFERENCES source_refs(id),
    confidence      REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(type);

-- Relationship properties (versioned, same pattern as entity properties)
CREATE TABLE IF NOT EXISTS relationship_properties (
    id                  TEXT PRIMARY KEY,
    relationship_id     TEXT NOT NULL REFERENCES relationships(id) ON DELETE CASCADE,
    key                 TEXT NOT NULL,
    value_json          TEXT,
    as_of               TEXT NOT NULL,
    recorded_at         TEXT NOT NULL,
    source_ref_id       TEXT REFERENCES source_refs(id),
    confidence          REAL NOT NULL DEFAULT 1.0,
    superseded_by_id    TEXT REFERENCES relationship_properties(id)
);

CREATE INDEX IF NOT EXISTS idx_rel_props_rel_key ON relationship_properties(relationship_id, key);

-- Merge log (audit trail for entity merges)
CREATE TABLE IF NOT EXISTS merge_log (
    id              TEXT PRIMARY KEY,
    survivor_id     TEXT NOT NULL REFERENCES entities(id),
    absorbed_id     TEXT NOT NULL,    -- may no longer exist after hard delete
    timestamp       TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    undo_data       TEXT NOT NULL DEFAULT '{}'  -- JSON blob for reversibility
);

CREATE INDEX IF NOT EXISTS idx_merge_log_survivor ON merge_log(survivor_id);

-- Conflicts (unresolved contradictions)
CREATE TABLE IF NOT EXISTS conflicts (
    id              TEXT PRIMARY KEY,
    entity_id       TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    property_key    TEXT NOT NULL,
    value_a_id      TEXT NOT NULL REFERENCES properties(id),
    value_b_id      TEXT NOT NULL REFERENCES properties(id),
    status          TEXT NOT NULL DEFAULT 'unresolved'
                    CHECK(status IN ('unresolved','resolved','auto_resolved')),
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    resolution      TEXT
);

CREATE INDEX IF NOT EXISTS idx_conflicts_entity ON conflicts(entity_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""


def _migrate_v1_to_v2(db: Database) -> None:
    """Add tier-counter columns and entity_sources table to a v1 database."""
    cols = {row[1] for row in db.fetchall("PRAGMA table_info(entities)")}
    if "mention_count" not in cols:
        db.execute("ALTER TABLE entities ADD COLUMN mention_count INTEGER NOT NULL DEFAULT 0")
    if "source_count" not in cols:
        db.execute("ALTER TABLE entities ADD COLUMN source_count INTEGER NOT NULL DEFAULT 0")
    if "tier" not in cols:
        db.execute("ALTER TABLE entities ADD COLUMN tier INTEGER NOT NULL DEFAULT 3")
    db.execute("CREATE INDEX IF NOT EXISTS idx_entities_tier ON entities(tier)")
    db.execute(
        """CREATE TABLE IF NOT EXISTS entity_sources (
            entity_id       TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            conversation_id TEXT NOT NULL,
            first_seen      TEXT NOT NULL,
            PRIMARY KEY (entity_id, conversation_id)
        )"""
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_entity_sources_conv ON entity_sources(conversation_id)")


def create_tables(db: Database) -> None:
    """Create all Memento tables if they don't exist. Idempotent.

    Also upgrades a v1 database in place to the current SCHEMA_VERSION
    so existing users keep working after an upgrade.
    """
    for statement in DDL.split(";"):
        statement = statement.strip()
        if statement:
            db.execute(statement)
    db.conn.commit()

    row = db.fetchone("SELECT version FROM schema_version LIMIT 1")
    if row is None:
        db.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
        )
        db.conn.commit()
        return

    current_version = row[0]
    if current_version < 2:
        _migrate_v1_to_v2(db)
        db.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        db.conn.commit()
