from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pwc_support.domain.models import CaseStatus, DeliveryReceipt, ReviewDecisionKind, ReviewDecisionResult
from pwc_support.storage.repositories import CaseRepository, MailboxRepository, OutboxRepository, ReviewRepository


class ReviewService:
    def __init__(self, reviews: ReviewRepository, dispatcher: OutboxDispatcher | None = None) -> None:
        self.reviews = reviews
        self.dispatcher = dispatcher

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
        if kind is ReviewDecisionKind.SEND_RESPONSE and not (reviewed_text or "").strip():
            raise ValueError("reviewed_text is required when sending a response")
        result = self.reviews.decide(
            review_id=review_id,
            decision_id=decision_id,
            expected_version=expected_version,
            reviewer_id=reviewer_id,
            kind=kind,
            reviewed_text=reviewed_text,
            reason=reason,
        )
        if self.dispatcher is not None and result.outbox_key:
            self.dispatcher.dispatch_once(worker_id="review-service")
        return result


@dataclass(slots=True)
class OutboxDispatcher:
    outbox: OutboxRepository
    mailbox: MailboxRepository
    cases: CaseRepository | None = None

    def dispatch_once(self, *, worker_id: str) -> DeliveryReceipt | None:
        from datetime import UTC, datetime

        message = self.outbox.claim_next(worker_id=worker_id, now=datetime.now(UTC))
        if message is None:
            return None
        try:
            receipt = self.mailbox.deliver(message)
        except Exception as exc:
            self.outbox.mark_failed(message.delivery_key, str(exc))
            raise
        self.outbox.mark_sent(message.delivery_key, receipt)
        if self.cases is not None and message.case_id:
            self.cases.update_status(message.case_id, CaseStatus.RESOLVED)
        return receipt
