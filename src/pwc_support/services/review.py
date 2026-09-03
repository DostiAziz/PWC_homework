from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pwc_support.domain.models import CaseStatus, DeliveryReceipt, ReviewDecisionKind, ReviewDecisionResult
from pwc_support.storage.repositories import CaseRepository, MailboxRepository, OutboxRepository, ReviewRepository
from pwc_support.workflow.retail_actions import RetailApprovalService


class ReviewService:
    def __init__(self, reviews: ReviewRepository, dispatcher: OutboxDispatcher | None = None, retail_approvals: RetailApprovalService | None = None, allowed_reviewer_ids: frozenset[str] | None = None) -> None:
        self.reviews = reviews
        self.dispatcher = dispatcher
        self.retail_approvals = retail_approvals
        self.allowed_reviewer_ids = allowed_reviewer_ids

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
        if self.allowed_reviewer_ids is not None and reviewer_id not in self.allowed_reviewer_ids:
            raise PermissionError("reviewer is not authorized to decide this case")
        if kind is ReviewDecisionKind.SEND_RESPONSE and not (reviewed_text or "").strip():
            raise ValueError("reviewed_text is required when sending a response")
        review = self.reviews.find(review_id)
        if review is None:
            raise KeyError(review_id)
        result = self.reviews.decide(
            review_id=review_id,
            decision_id=decision_id,
            expected_version=expected_version,
            reviewer_id=reviewer_id,
            kind=kind,
            reviewed_text=reviewed_text,
            reason=reason,
        )
        if not result.replayed and self.retail_approvals is not None and kind in {ReviewDecisionKind.APPROVE_REFUND, ReviewDecisionKind.REJECT_REFUND, ReviewDecisionKind.APPROVE_RETURN}:
            for action in review.proposed_actions:
                if action.action_type == "retail_return":
                    self.retail_approvals.apply(action=action.model_dump(mode="json"), kind=kind.value)
                    break
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
