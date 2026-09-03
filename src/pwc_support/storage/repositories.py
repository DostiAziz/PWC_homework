from __future__ import annotations

import json
import sqlite3
from uuid import UUID

from pwc_support.domain.models import (
    CaseRecord,
    CaseRequest,
    CaseStatus,
    DraftReply,
    ProposedAction,
    ReviewRequest,
)
from pwc_support.storage.database import Database, json_object


class CaseRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, request: CaseRequest, *, case_id: str) -> CaseRecord:
        record = CaseRecord(
            case_id=case_id,
            conversation_id=request.conversation_id,
            client_id=request.client_id,
            category=request.category,
            status=CaseStatus.PENDING_REVIEW,
            summary=request.summary,
            version=1,
        )
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO cases
                (case_id, conversation_id, client_id, category, status, summary, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.case_id,
                    str(record.conversation_id),
                    record.client_id,
                    record.category,
                    record.status.value,
                    record.summary,
                    record.version,
                ),
            )
        return self.get(case_id)

    def get(self, case_id: str) -> CaseRecord:
        record = self.find(case_id)
        if record is None:
            raise KeyError(case_id)
        return record

    def find(self, case_id: str) -> CaseRecord | None:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return None if row is None else self._from_row(row)

    def update_status(self, case_id: str, status: CaseStatus) -> CaseRecord:
        """Advance a case and bump its optimistic-concurrency version."""
        with self.database.connect() as connection:
            updated = connection.execute(
                "UPDATE cases SET status = ?, version = version + 1 WHERE case_id = ?",
                (status.value, case_id),
            ).rowcount
        if not updated:
            raise KeyError(case_id)
        return self.get(case_id)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> CaseRecord:
        return CaseRecord(
            case_id=row["case_id"],
            conversation_id=UUID(row["conversation_id"]),
            client_id=row["client_id"],
            category=row["category"],
            status=CaseStatus(row["status"]),
            summary=row["summary"],
            version=row["version"],
        )


class ReviewRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create(self, review: ReviewRequest) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO review_requests
                (review_id, case_id, run_id, categories_json, original_message,
                 proposed_reply_json, proposed_actions_json, response_version, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(review.review_id),
                    review.case_id,
                    str(review.run_id),
                    json_object([category.value for category in review.categories]),
                    review.original_message,
                    review.proposed_reply.model_dump_json() if review.proposed_reply else None,
                    json_object([action.model_dump() for action in review.proposed_actions]),
                    review.response_version,
                    review.status,
                ),
            )

    def list_pending(self) -> list[ReviewRequest]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM review_requests WHERE status = 'pending' ORDER BY rowid"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def find(self, review_id: UUID) -> ReviewRequest | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_requests WHERE review_id = ?", (str(review_id),)
            ).fetchone()
        return None if row is None else self._from_row(row)

    def mark_decided(self, review_id: UUID) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE review_requests SET status = 'decided' WHERE review_id = ?",
                (str(review_id),),
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ReviewRequest:
        proposed_reply_json = row["proposed_reply_json"]
        return ReviewRequest(
            review_id=UUID(row["review_id"]),
            case_id=row["case_id"],
            run_id=UUID(row["run_id"]),
            categories=frozenset(json.loads(row["categories_json"])),
            original_message=row["original_message"],
            proposed_reply=(
                DraftReply.model_validate_json(proposed_reply_json)
                if proposed_reply_json
                else None
            ),
            proposed_actions=tuple(
                ProposedAction.model_validate(item)
                for item in json.loads(row["proposed_actions_json"])
            ),
            response_version=row["response_version"],
            status=row["status"],
        )
