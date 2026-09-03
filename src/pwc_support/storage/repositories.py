from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from pwc_support.domain.errors import ErrorCode, SupportError
from pwc_support.domain.models import (
    CaseRecord,
    CaseRequest,
    CaseStatus,
    Citation,
    DeliveryReceipt,
    DraftReply,
    InboundClaim,
    InboundClaimResult,
    MessageKind,
    OutboundMessage,
    ProposedAction,
    ReviewDecisionKind,
    ReviewDecisionResult,
    ReviewRequest,
    RoutingSnapshot,
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
                 proposed_reply_json, proposed_actions_json, evidence_json, checkpoint_id,
                 response_version, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(review.review_id),
                    review.case_id,
                    str(review.run_id),
                    json_object([category.value for category in review.categories]),
                    review.original_message,
                    review.proposed_reply.model_dump_json() if review.proposed_reply else None,
                    json_object([action.model_dump() for action in review.proposed_actions]),
                    json_object([citation.model_dump(mode="json") for citation in review.evidence]),
                    review.checkpoint_id,
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
        evidence_json = row["evidence_json"]
        return ReviewRequest(
            review_id=UUID(row["review_id"]),
            case_id=row["case_id"],
            run_id=UUID(row["run_id"]),
            categories=frozenset(json.loads(row["categories_json"])),
            original_message=row["original_message"],
            proposed_reply=(
                DraftReply.model_validate_json(proposed_reply_json) if proposed_reply_json else None
            ),
            proposed_actions=tuple(
                ProposedAction.model_validate(item)
                for item in json.loads(row["proposed_actions_json"])
            ),
            evidence=tuple(
                Citation.model_validate(item) for item in json.loads(evidence_json or "[]")
            ),
            checkpoint_id=row["checkpoint_id"],
            response_version=row["response_version"],
            status=row["status"],
        )

    def decide(
        self,
        *,
        review_id: UUID,
        decision_id: UUID,
        expected_version: int,
        reviewer_id: str,
        kind: ReviewDecisionKind,
        reviewed_text: str | None = None,
        reason: str | None = None,
    ) -> ReviewDecisionResult:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_requests WHERE review_id = ?", (str(review_id),)
            ).fetchone()
            if row is None:
                raise KeyError(review_id)
            existing = connection.execute(
                "SELECT * FROM review_decisions WHERE review_id = ? AND expected_version = ?",
                (str(review_id), expected_version),
            ).fetchone()
            if existing is not None:
                return ReviewDecisionResult(
                    decision_id=UUID(existing["decision_id"]),
                    review_id=review_id,
                    kind=ReviewDecisionKind(existing["kind"]),
                    case_status=CaseStatus.RESOLVED,
                    outbox_key=f"case:{row['case_id']}:response:{expected_version}",
                    replayed=True,
                )
            if row["status"] != "pending" or row["response_version"] != expected_version:
                raise SupportError(
                    ErrorCode.VERSION_CONFLICT, "review is no longer at the expected version"
                )
            now = datetime.now(UTC).isoformat()
            outbox_key = None
            case_status = (
                CaseStatus.RESOLVED
                if kind in (ReviewDecisionKind.SEND_RESPONSE, ReviewDecisionKind.APPROVE)
                else CaseStatus.REJECTED
            )
            if kind in (ReviewDecisionKind.SEND_RESPONSE, ReviewDecisionKind.APPROVE):
                outbox_key = f"case:{row['case_id']}:response:{expected_version}"
                body = reviewed_text or "Your request has been reviewed by our support team."
                payload_hash = str(hash(body))
                connection.execute(
                    "INSERT OR IGNORE INTO outbox_messages (delivery_key,case_id,review_id,response_version,recipient,thread_id,subject,body,message_kind,payload_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        outbox_key,
                        row["case_id"],
                        str(review_id),
                        expected_version,
                        "client@example.test",
                        f"case:{row['case_id']}",
                        "Support response",
                        body,
                        MessageKind.REVIEWED_RESPONSE.value,
                        payload_hash,
                        now,
                        now,
                    ),
                )
            connection.execute(
                "INSERT INTO review_decisions (decision_id,review_id,expected_version,kind,reviewer_id,reviewed_text,reason,content_hash,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(decision_id),
                    str(review_id),
                    expected_version,
                    kind.value,
                    reviewer_id,
                    reviewed_text,
                    reason,
                    str(hash(reviewed_text or "")),
                    now,
                ),
            )
            connection.execute(
                "UPDATE review_requests SET status='decided', updated_at=? WHERE review_id=? AND status='pending'",
                (now, str(review_id)),
            )
            connection.execute(
                "UPDATE cases SET status=?, version=version+1, updated_at=? WHERE case_id=?",
                (case_status.value, now, row["case_id"]),
            )
        return ReviewDecisionResult(
            decision_id=decision_id,
            review_id=review_id,
            kind=kind,
            case_status=case_status,
            outbox_key=outbox_key,
        )


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class InboundRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def claim(
        self,
        provider: str,
        provider_message_id: str,
        payload_hash: str,
        *,
        lease_seconds: int,
        now: datetime,
    ) -> InboundClaimResult:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT * FROM inbound_messages WHERE provider=? AND provider_message_id=?",
                (provider, provider_message_id),
            ).fetchone()
            if row is None:
                token = str(uuid4())
                expiry = now + timedelta(seconds=lease_seconds)
                c.execute(
                    "INSERT INTO inbound_messages (provider,provider_message_id,payload_hash,message_json,conversation_id,provider_thread_id,sender_id,body,status,claim_token,lease_owner,lease_expires_at,attempt_count,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        provider,
                        provider_message_id,
                        payload_hash,
                        "{}",
                        provider_message_id,
                        provider_message_id,
                        "unknown",
                        "",
                        "processing",
                        token,
                        token,
                        expiry.isoformat(),
                        1,
                        now.isoformat(),
                    ),
                )
                return InboundClaimResult(
                    status="claimed",
                    claim=InboundClaim(
                        provider=provider,
                        provider_message_id=provider_message_id,
                        payload_hash=payload_hash,
                        claim_token=token,
                        lease_expires_at=expiry,
                        attempt_count=1,
                    ),
                )
            if row["payload_hash"] != payload_hash:
                raise SupportError(
                    ErrorCode.IDEMPOTENCY_CONFLICT, "payload hash conflicts with existing message"
                )
            if row["status"] == "completed":
                return InboundClaimResult(status="completed")
            existing_expiry = _dt(row["lease_expires_at"])
            if existing_expiry and existing_expiry > now:
                return InboundClaimResult(status="in_progress")
            token = str(uuid4())
            new_expiry = now + timedelta(seconds=lease_seconds)
            c.execute(
                "UPDATE inbound_messages SET claim_token=?, lease_owner=?, lease_expires_at=?, attempt_count=attempt_count+1, status='processing' WHERE inbound_id=?",
                (token, token, new_expiry.isoformat(), row["inbound_id"]),
            )
            return InboundClaimResult(
                status="claimed",
                claim=InboundClaim(
                    provider=provider,
                    provider_message_id=provider_message_id,
                    payload_hash=payload_hash,
                    claim_token=token,
                    lease_expires_at=new_expiry,
                    attempt_count=row["attempt_count"] + 1,
                ),
            )

    def record_routing_snapshot(
        self, claim: InboundClaim, snapshot: RoutingSnapshot
    ) -> RoutingSnapshot:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT * FROM inbound_messages WHERE provider=? AND provider_message_id=?",
                (claim.provider, claim.provider_message_id),
            ).fetchone()
            if row is None or row["claim_token"] != claim.claim_token:
                raise SupportError(ErrorCode.STALE_INBOUND_CLAIM, "inbound claim is stale")
            if row["routing_snapshot_json"]:
                return RoutingSnapshot.model_validate_json(row["routing_snapshot_json"])
            c.execute(
                "UPDATE inbound_messages SET routing_snapshot_json=?, routing_snapshot_hash=? WHERE inbound_id=? AND claim_token=?",
                (
                    snapshot.model_dump_json(),
                    snapshot.snapshot_hash,
                    row["inbound_id"],
                    claim.claim_token,
                ),
            )
            return snapshot

    def renew(self, claim: InboundClaim, *, lease_seconds: int, now: datetime) -> InboundClaim:
        with self.database.connect() as c:
            expiry = now + timedelta(seconds=lease_seconds)
            updated = c.execute(
                "UPDATE inbound_messages SET lease_expires_at=? WHERE provider=? AND provider_message_id=? AND claim_token=?",
                (expiry.isoformat(), claim.provider, claim.provider_message_id, claim.claim_token),
            ).rowcount
            if not updated:
                raise SupportError(ErrorCode.STALE_INBOUND_CLAIM, "inbound claim is stale")
        return claim.model_copy(update={"lease_expires_at": expiry})


class OutboxRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def insert(self, message: OutboundMessage) -> None:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT payload_hash FROM outbox_messages WHERE delivery_key=?",
                (message.delivery_key,),
            ).fetchone()
            if row and row["payload_hash"] != message.payload_hash:
                raise SupportError(ErrorCode.IDEMPOTENCY_CONFLICT, "delivery key payload conflict")
            now = datetime.now(UTC).isoformat()
            c.execute(
                "INSERT OR IGNORE INTO outbox_messages (delivery_key,case_id,response_version,recipient,thread_id,subject,body,message_kind,payload_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    message.delivery_key,
                    message.case_id,
                    message.response_version,
                    message.recipient,
                    message.thread_id,
                    message.subject,
                    message.body,
                    message.kind.value,
                    message.payload_hash,
                    now,
                    now,
                ),
            )

    def claim_next(
        self, *, worker_id: str, now: datetime, lease_seconds: int = 60
    ) -> OutboundMessage | None:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT * FROM outbox_messages WHERE status='pending' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            c.execute(
                "UPDATE outbox_messages SET status='processing',worker_id=?,lease_expires_at=?,attempt_count=attempt_count+1 WHERE delivery_key=?",
                (
                    worker_id,
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    row["delivery_key"],
                ),
            )
            return OutboundMessage(
                delivery_key=row["delivery_key"],
                kind=MessageKind(row["message_kind"]),
                case_id=row["case_id"],
                response_version=row["response_version"],
                recipient=row["recipient"],
                thread_id=row["thread_id"],
                subject=row["subject"],
                body=row["body"],
                payload_hash=row["payload_hash"],
            )

    def mark_sent(self, delivery_key: str, receipt: DeliveryReceipt) -> None:
        with self.database.connect() as c:
            c.execute("UPDATE outbox_messages SET status='sent', provider_message_id=?, updated_at=? WHERE delivery_key=?", (receipt.provider_message_id, receipt.delivered_at.isoformat(), delivery_key))

    def mark_failed(self, delivery_key: str, error: str) -> None:
        with self.database.connect() as c:
            c.execute("UPDATE outbox_messages SET status='failed', last_error=?, updated_at=? WHERE delivery_key=?", (error, datetime.now(UTC).isoformat(), delivery_key))

    def retry(self, delivery_key: str) -> None:
        with self.database.connect() as c:
            c.execute("UPDATE outbox_messages SET status='pending', updated_at=? WHERE delivery_key=? AND status IN ('failed','processing')", (datetime.now(UTC).isoformat(), delivery_key))


class MailboxRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def deliver(self, message: OutboundMessage) -> DeliveryReceipt:
        with self.database.connect() as c:
            row = c.execute(
                "SELECT * FROM mailbox_messages WHERE delivery_key=?", (message.delivery_key,)
            ).fetchone()
            if row:
                if row["payload_hash"] != message.payload_hash:
                    raise SupportError(
                        ErrorCode.IDEMPOTENCY_CONFLICT, "delivery key payload conflict"
                    )
                return DeliveryReceipt(
                    delivery_key=message.delivery_key,
                    provider_message_id=row["provider_message_id"]
                    or f"mail-{row['mailbox_message_id']}",
                    delivered_at=_dt(row["created_at"]) or datetime.now(UTC),
                )
            now = datetime.now(UTC)
            provider_id = f"mail-{uuid4()}"
            c.execute(
                "INSERT INTO mailbox_messages (delivery_key,provider_message_id,provider_thread_id,sender_id,recipient_id,subject,body,message_kind,payload_hash,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    message.delivery_key,
                    provider_id,
                    message.thread_id,
                    "support",
                    message.recipient,
                    message.subject,
                    message.body,
                    message.kind.value,
                    message.payload_hash,
                    now.isoformat(),
                ),
            )
            return DeliveryReceipt(
                delivery_key=message.delivery_key, provider_message_id=provider_id, delivered_at=now
            )

    def list_messages(self, *, recipient: str, thread_id: str) -> list[dict[str, object]]:
        with self.database.connect() as c:
            rows = c.execute("SELECT * FROM mailbox_messages WHERE recipient_id=? AND provider_thread_id=? ORDER BY created_at", (recipient, thread_id)).fetchall()
        return [dict(row) for row in rows]
