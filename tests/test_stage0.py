"""Stage 0 tests: project skeleton, config, DB connection, embedder protocol."""

import sqlite3
import tempfile
from pathlib import Path

from memento import __version__
from memento.config import EmbeddingConfig, LLMConfig, MementoConfig
from memento.db import Database


def test_version():
    assert __version__ == "0.1.0"


def test_import_all_modules():
    import memento.config
    import memento.db
    import memento.embedder


def test_default_config():
    config = MementoConfig()
    assert config.embedding.dimension == 384
    assert config.embedding.provider == "auto"
    assert config.llm.extraction_model == ""  # Auto-set from provider at runtime
    assert config.resolution.high_threshold == 0.85
    assert config.resolution.low_threshold == 0.40
    assert config.retrieval.default_token_budget == 2000


def test_sqlite_wal_mode(db):
    # In-memory databases use "memory" journal mode; WAL only applies to file-based DBs.
    # Verify the PRAGMA was issued without error (the connection works).
    result = db.fetchone("PRAGMA journal_mode")
    assert result[0] in ("wal", "memory")


def test_sqlite_foreign_keys(db):
    result = db.fetchone("PRAGMA foreign_keys")
    assert result[0] == 1


def test_transaction_commit(db):
    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
    with db.transaction() as cur:
        cur.execute("INSERT INTO test (val) VALUES (?)", ("hello",))
    row = db.fetchone("SELECT val FROM test WHERE id = 1")
    assert row["val"] == "hello"


def test_transaction_rollback(db):
    db.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
    db.execute("INSERT INTO test (val) VALUES (?)", ("original",))
    db.conn.commit()

    try:
        with db.transaction() as cur:
            cur.execute("UPDATE test SET val = ? WHERE id = 1", ("modified",))
            raise ValueError("simulated failure")
    except ValueError:
        pass

    row = db.fetchone("SELECT val FROM test WHERE id = 1")
    assert row["val"] == "original"


def test_database_context_manager():
    with Database(":memory:") as db:
        result = db.fetchone("PRAGMA journal_mode")
        assert result[0] in ("wal", "memory")


def test_sqlite_wal_mode_file_based(tmp_path):
    """WAL mode should activate on file-based databases."""
    db_path = tmp_path / "test.db"
    with Database(db_path) as db:
        result = db.fetchone("PRAGMA journal_mode")
        assert result[0] == "wal"


def test_embedding_config_defaults():
    config = EmbeddingConfig()
    assert config.dimension == 384
    assert config.model == "all-MiniLM-L6-v2"
    assert config.provider == "auto"


def test_embedder_protocol():
    """Verify the Embedder protocol is importable and has the right shape."""
    from memento.embedder import Embedder

    # Protocol should define these attributes
    assert hasattr(Embedder, "dimension")
    assert hasattr(Embedder, "embed")
    assert hasattr(Embedder, "embed_batch")
