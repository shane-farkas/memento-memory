"""SQLite connection management for Memento."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


class Database:
    """Manages a SQLite connection with WAL mode and transaction helpers."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        logger.debug("Opened SQLite connection: %s (WAL mode)", self.db_path)
        return conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for atomic transactions.

        Usage:
            with db.transaction() as cur:
                cur.execute("INSERT INTO ...")
                cur.execute("UPDATE ...")
            # auto-commits on exit, rolls back on exception
        """
        conn = self.conn
        cur = conn.cursor()
        try:
            cur.execute("BEGIN")
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq: list[tuple]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params_seq)

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Closed SQLite connection: %s", self.db_path)

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
