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
                    version INTEGER NOT NULL,
                    inbound_message_id INTEGER,
                    reply_recipient TEXT,
                    delivery_thread_id TEXT,
                    assigned_reviewer TEXT,
                    created_at TEXT,
                    updated_at TEXT
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
                    status TEXT NOT NULL,
                    inbound_message_id INTEGER,
                    evidence_state TEXT,
                    routing_provenance_json TEXT,
                    delivery_recipient TEXT,
                    delivery_thread_id TEXT,
                    delivery_subject TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS inbound_messages (
                    inbound_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    provider_message_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    provider_thread_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT,
                    subject TEXT,
                    body TEXT NOT NULL,
                    run_id TEXT,
                    status TEXT NOT NULL DEFAULT 'processing',
                    claim_token TEXT,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    routing_snapshot_json TEXT,
                    routing_snapshot_hash TEXT,
                    final_routing_json TEXT,
                    outcome_json TEXT,
                    case_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(provider, provider_message_id)
                );
                CREATE TABLE IF NOT EXISTS review_decisions (
                    decision_id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL,
                    expected_version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    reviewed_text TEXT,
                    reason TEXT,
                    content_hash TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(review_id, expected_version)
                );
                CREATE TABLE IF NOT EXISTS outbox_messages (
                    delivery_key TEXT PRIMARY KEY,
                    case_id TEXT,
                    review_id TEXT,
                    response_version INTEGER,
                    recipient TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    message_kind TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    lease_expires_at TEXT,
                    last_error TEXT,
                    provider_message_id TEXT,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mailbox_messages (
                    mailbox_message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    delivery_key TEXT NOT NULL UNIQUE,
                    conversation_id TEXT,
                    provider_message_id TEXT,
                    provider_thread_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    message_kind TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
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
                CREATE INDEX IF NOT EXISTS idx_inbound_lease
                    ON inbound_messages(status, lease_expires_at);
                CREATE INDEX IF NOT EXISTS idx_outbox_status
                    ON outbox_messages(status, lease_expires_at);
                CREATE INDEX IF NOT EXISTS idx_mailbox_thread
                    ON mailbox_messages(recipient_id, provider_thread_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_review_case_version
                    ON review_requests(case_id, response_version);
                """
            )
            retail_schema = (
                Path(__file__).with_name("retail_schema.sql").read_text(encoding="utf-8")
            )
            connection.executescript(retail_schema)
            # `CREATE TABLE IF NOT EXISTS` leaves a database made by an earlier version
            # untouched, so columns added later are backfilled explicitly.
            self._add_missing_column(connection, "review_requests", "evidence_json", "TEXT")
            self._add_missing_column(connection, "review_requests", "checkpoint_id", "TEXT")
            for column in (
                "inbound_message_id",
                "reply_recipient",
                "delivery_thread_id",
                "assigned_reviewer",
                "created_at",
                "updated_at",
            ):
                self._add_missing_column(connection, "cases", column, "TEXT")
            for column in (
                "inbound_message_id",
                "evidence_state",
                "routing_provenance_json",
                "delivery_recipient",
                "delivery_thread_id",
                "delivery_subject",
                "created_at",
                "updated_at",
            ):
                self._add_missing_column(connection, "review_requests", column, "TEXT")
            self._add_missing_column(
                connection, "return_requests", "version", "INTEGER NOT NULL DEFAULT 1"
            )
            self._add_missing_column(
                connection, "refund_requests", "version", "INTEGER NOT NULL DEFAULT 1"
            )
            for column, declaration in (
                ("payment_status", "TEXT NOT NULL DEFAULT 'unpaid'"),
                ("fulfilment_status", "TEXT NOT NULL DEFAULT 'processing'"),
                ("shipped_at", "TEXT"),
                ("cancelled_at", "TEXT"),
                ("version", "INTEGER NOT NULL DEFAULT 1"),
            ):
                self._add_missing_column(connection, "orders", column, declaration)
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_cases_inbound_message
                ON cases(inbound_message_id)
                WHERE inbound_message_id IS NOT NULL
                """
            )

    @staticmethod
    def _add_missing_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        existing = {
            str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
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
