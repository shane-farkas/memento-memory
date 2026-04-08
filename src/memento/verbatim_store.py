"""Verbatim text storage with hybrid search (vector + FTS5)."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field

import numpy as np

from memento.db import Database
from memento.embedder import Embedder
from memento.models import _new_id, _now

logger = logging.getLogger(__name__)

VERBATIM_DDL = """
-- Raw conversation text chunks
CREATE TABLE IF NOT EXISTS verbatim_chunks (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL DEFAULT '',
    turn_number     INTEGER NOT NULL DEFAULT 0,
    text            TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    participant     TEXT NOT NULL DEFAULT '',
    source_type     TEXT NOT NULL DEFAULT 'conversation'
);

CREATE INDEX IF NOT EXISTS idx_verbatim_conversation ON verbatim_chunks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_verbatim_timestamp ON verbatim_chunks(timestamp);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS verbatim_fts USING fts5(
    text,
    content=verbatim_chunks,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS verbatim_fts_insert AFTER INSERT ON verbatim_chunks BEGIN
    INSERT INTO verbatim_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


@dataclass
class VerbatimResult:
    """A single result from verbatim search."""

    chunk_id: str
    text: str
    conversation_id: str
    turn_number: int
    timestamp: str
    participant: str
    score: float = 0.0


def _float_array_to_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array to little-endian float32 bytes for sqlite-vec."""
    return arr.astype(np.float32).tobytes()


def _bytes_to_float_array(b: bytes, dim: int) -> np.ndarray:
    """Convert bytes back to numpy array."""
    return np.frombuffer(b, dtype=np.float32)


class VerbatimStore:
    """Stores raw conversation text with vector + FTS5 hybrid search."""

    def __init__(self, db: Database, embedder: Embedder) -> None:
        self.db = db
        self.embedder = embedder
        self._init_tables()

    def _init_tables(self) -> None:
        """Create verbatim tables and vector index."""
        import sqlite_vec

        self.db.conn.enable_load_extension(True)
        sqlite_vec.load(self.db.conn)
        self.db.conn.enable_load_extension(False)

        # Use executescript for DDL that contains triggers with semicolons
        self.db.conn.executescript(VERBATIM_DDL)

        # Create vec0 virtual table for vector search
        dim = self.embedder.dimension
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS verbatim_vec USING vec0(embedding float[{dim}])"
        )
        self.db.conn.commit()

        # Map chunk IDs to rowids for the vec table
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS verbatim_vec_map (
                chunk_id TEXT PRIMARY KEY,
                vec_rowid INTEGER NOT NULL
            )"""
        )
        self.db.conn.commit()

    def store(
        self,
        text: str,
        conversation_id: str = "",
        turn_number: int = 0,
        participant: str = "",
        source_type: str = "conversation",
        timestamp: str | None = None,
    ) -> str:
        """Store a verbatim text chunk with its embedding. Returns chunk ID."""
        chunk_id = _new_id()
        ts = timestamp or _now()

        # Insert the text chunk
        self.db.execute(
            """INSERT INTO verbatim_chunks (id, conversation_id, turn_number, text, timestamp, participant, source_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chunk_id, conversation_id, turn_number, text, ts, participant, source_type),
        )

        # Get the rowid
        row = self.db.fetchone(
            "SELECT rowid FROM verbatim_chunks WHERE id = ?", (chunk_id,)
        )
        rowid = row[0]

        # Compute and store embedding
        embedding = self.embedder.embed(text)
        embedding_bytes = _float_array_to_bytes(embedding)

        self.db.execute(
            "INSERT INTO verbatim_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, embedding_bytes),
        )

        # Map chunk_id to vec rowid
        self.db.execute(
            "INSERT INTO verbatim_vec_map (chunk_id, vec_rowid) VALUES (?, ?)",
            (chunk_id, rowid),
        )

        self.db.conn.commit()
        logger.debug("Stored verbatim chunk: %s (%d chars)", chunk_id[:8], len(text))
        return chunk_id

    def search(self, query: str, top_k: int = 10) -> list[VerbatimResult]:
        """Hybrid search: vector similarity + BM25 keyword matching.

        Uses reciprocal rank fusion to combine results.
        """
        vec_results = self._vector_search(query, top_k)
        fts_results = self._fts_search(query, top_k)
        return self._fuse_results(vec_results, fts_results, top_k)

    def get_by_conversation(self, conversation_id: str) -> list[VerbatimResult]:
        """Get all verbatim chunks from a specific conversation."""
        rows = self.db.fetchall(
            """SELECT * FROM verbatim_chunks
               WHERE conversation_id = ?
               ORDER BY turn_number ASC""",
            (conversation_id,),
        )
        return [self._row_to_result(row, score=1.0) for row in rows]

    def _vector_search(self, query: str, top_k: int) -> list[VerbatimResult]:
        """Search by embedding similarity."""
        query_embedding = self.embedder.embed(query)
        query_bytes = _float_array_to_bytes(query_embedding)

        rows = self.db.fetchall(
            """SELECT v.rowid, v.distance
               FROM verbatim_vec v
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance ASC""",
            (query_bytes, top_k),
        )

        results = []
        for row in rows:
            # Look up the chunk by rowid
            chunk_row = self.db.fetchone(
                "SELECT * FROM verbatim_chunks WHERE rowid = ?", (row["rowid"],)
            )
            if chunk_row:
                # Convert distance to similarity score (0=identical → 1.0 score)
                score = 1.0 / (1.0 + row["distance"])
                results.append(self._row_to_result(chunk_row, score=score))
        return results

    def _fts_search(self, query: str, top_k: int) -> list[VerbatimResult]:
        """Search by BM25 keyword matching."""
        # Escape FTS5 special characters
        safe_query = query.replace('"', '""')

        try:
            rows = self.db.fetchall(
                f"""SELECT c.*, bm25(verbatim_fts) as rank
                    FROM verbatim_fts f
                    JOIN verbatim_chunks c ON c.rowid = f.rowid
                    WHERE verbatim_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?""",
                (f'"{safe_query}"', top_k),
            )
        except Exception:
            # FTS query might fail on complex queries; fall back to empty
            return []

        results = []
        for row in rows:
            # BM25 returns negative scores (more negative = more relevant)
            score = 1.0 / (1.0 + abs(row["rank"]))
            results.append(self._row_to_result(row, score=score))
        return results

    def _fuse_results(
        self,
        vec_results: list[VerbatimResult],
        fts_results: list[VerbatimResult],
        top_k: int,
    ) -> list[VerbatimResult]:
        """Reciprocal rank fusion of vector and FTS results."""
        k = 60  # RRF constant

        scores: dict[str, float] = {}
        result_map: dict[str, VerbatimResult] = {}

        for rank, r in enumerate(vec_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(fts_results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        # Sort by fused score descending
        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            r = result_map[cid]
            r.score = scores[cid]
            results.append(r)
        return results

    def _row_to_result(self, row, score: float = 0.0) -> VerbatimResult:
        return VerbatimResult(
            chunk_id=row["id"],
            text=row["text"],
            conversation_id=row["conversation_id"],
            turn_number=row["turn_number"],
            timestamp=row["timestamp"],
            participant=row["participant"],
            score=score,
        )
