from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from uuid import UUID

from pwc_support.domain.models import ConversationMemory, SpecialistName
from pwc_support.storage.database import Database

# Mirrors Settings.conversation_memory_max_chars; storage must not import from
# pwc_support.config so callers pass the configured limit explicitly.
DEFAULT_MAX_STATE_CHARS = 12000


class ConversationStateRepository:
    """Durable, per-(conversation, client) specialist memory used to resume clarification loops.

    Rows are scoped by the composite primary key `(conversation_id, client_id)`, so
    `load`/`save`/`clear` never expose one client's memory to another. `save` is a
    compare-and-swap: pass `expected_version=None` for the first write, then the
    version returned by the previous `load`/`save` for every later write.
    """

    def __init__(
        self, database: Database, *, max_state_chars: int = DEFAULT_MAX_STATE_CHARS
    ) -> None:
        self.database = database
        self.max_state_chars = max_state_chars
        # `Database.initialize()` is idempotent (CREATE TABLE IF NOT EXISTS plus
        # additive column backfills), so it is safe to call from every constructor
        # rather than requiring every caller to initialize the database up front.
        self.database.initialize()

    def load(self, conversation_id: UUID, client_id: str) -> ConversationMemory | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_agent_state "
                "WHERE conversation_id = ? AND client_id = ?",
                (str(conversation_id), client_id),
            ).fetchone()
        return None if row is None else self._from_row(row)

    def save(
        self, memory: ConversationMemory, *, expected_version: int | None
    ) -> ConversationMemory:
        payload = json.dumps(memory.state_json, default=str)
        if len(payload) > self.max_state_chars:
            raise ValueError(
                f"conversation memory exceeds {self.max_state_chars} characters"
            )
        now = datetime.now(UTC)
        new_version = 1 if expected_version is None else expected_version + 1
        specialist = memory.active_specialist.value if memory.active_specialist else None
        with self.database.connect() as connection:
            if expected_version is None:
                # A composite-PK conflict means a row already exists, which is a stale
                # write from the caller's point of view: treat it the same as a
                # version mismatch rather than silently overwriting another turn's state.
                updated = connection.execute(
                    """
                    INSERT INTO conversation_agent_state
                    (conversation_id, client_id, version, state_json, active_specialist,
                     active_task_id, updated_at)
                    SELECT ?, ?, ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM conversation_agent_state
                        WHERE conversation_id = ? AND client_id = ?
                    )
                    """,
                    (
                        str(memory.conversation_id),
                        memory.client_id,
                        new_version,
                        payload,
                        specialist,
                        memory.active_task_id,
                        now.isoformat(),
                        str(memory.conversation_id),
                        memory.client_id,
                    ),
                ).rowcount
            else:
                updated = connection.execute(
                    """
                    UPDATE conversation_agent_state
                    SET version = ?, state_json = ?, active_specialist = ?,
                        active_task_id = ?, updated_at = ?
                    WHERE conversation_id = ? AND client_id = ? AND version = ?
                    """,
                    (
                        new_version,
                        payload,
                        specialist,
                        memory.active_task_id,
                        now.isoformat(),
                        str(memory.conversation_id),
                        memory.client_id,
                        expected_version,
                    ),
                ).rowcount
        if not updated:
            raise ValueError("memory version conflict")
        return memory.model_copy(update={"version": new_version, "updated_at": now})

    def clear(self, conversation_id: UUID, client_id: str, *, expected_version: int) -> None:
        with self.database.connect() as connection:
            updated = connection.execute(
                "DELETE FROM conversation_agent_state "
                "WHERE conversation_id = ? AND client_id = ? AND version = ?",
                (str(conversation_id), client_id, expected_version),
            ).rowcount
        if not updated:
            raise ValueError("memory version conflict")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ConversationMemory:
        active_specialist = row["active_specialist"]
        return ConversationMemory(
            conversation_id=UUID(row["conversation_id"]),
            client_id=row["client_id"],
            active_specialist=SpecialistName(active_specialist) if active_specialist else None,
            active_task_id=row["active_task_id"],
            state_json=json.loads(row["state_json"]),
            version=row["version"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
