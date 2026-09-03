from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from pwc_support.domain.models import OperationalEvent


class Database:
    """Small SQLite boundary used by the prototype's application services."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_requests (
                    review_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    categories_json TEXT NOT NULL,
                    original_message TEXT NOT NULL,
                    proposed_reply_json TEXT,
                    proposed_actions_json TEXT NOT NULL,
                    evidence_json TEXT,
                    checkpoint_id TEXT,
                    response_version INTEGER NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operational_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    duration_ms REAL,
                    details_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_review_status ON review_requests(status);
                CREATE INDEX IF NOT EXISTS idx_event_run ON operational_events(run_id, event_id);
                """
            )
            # `CREATE TABLE IF NOT EXISTS` leaves a database made by an earlier version
            # untouched, so columns added later are backfilled explicitly.
            self._add_missing_column(connection, "review_requests", "evidence_json", "TEXT")
            self._add_missing_column(connection, "review_requests", "checkpoint_id", "TEXT")

    @staticmethod
    def _add_missing_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        existing = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def record_event(self, event: OperationalEvent) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO operational_events
                (run_id, conversation_id, node, event_type, duration_ms, details_json, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.run_id),
                    str(event.conversation_id),
                    event.node,
                    event.event_type,
                    event.duration_ms,
                    json.dumps(event.details),
                    event.occurred_at.isoformat(),
                ),
            )

    def list_events(self, run_id: UUID) -> list[OperationalEvent]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM operational_events WHERE run_id = ? ORDER BY event_id",
                (str(run_id),),
            ).fetchall()
        return [
            OperationalEvent(
                run_id=UUID(row["run_id"]),
                conversation_id=UUID(row["conversation_id"]),
                node=row["node"],
                event_type=row["event_type"],
                duration_ms=row["duration_ms"],
                details=json.loads(row["details_json"]),
                occurred_at=row["occurred_at"],
            )
            for row in rows
        ]


def json_object(value: Any) -> str:
    return json.dumps(value, default=str)
