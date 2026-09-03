from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from pwc_support.domain.models import ReturnEligibility, ReturnRequest, RetailActionRisk
from pwc_support.storage.retail_repositories import RefundRepository, ReturnRepository


def evaluate_return(
    repository: ReturnRepository,
    *,
    order_id: str,
    item_id: str,
    reason: str,
    now: datetime | None = None,
) -> ReturnEligibility:
    return repository.evaluate(order_id, item_id, reason, now or datetime.now(UTC))


def request_return(
    repository: ReturnRepository,
    *,
    order_id: str,
    item_id: str,
    reason: str,
    idempotency_key: str,
    now: datetime | None = None,
    auto_approval_limit: Decimal = Decimal("100.00"),
) -> tuple[ReturnRequest | None, ReturnEligibility]:
    eligibility = evaluate_return(repository, order_id=order_id, item_id=item_id, reason=reason, now=now)
    if not eligibility.eligible:
        return None, eligibility
    risk_flags = eligibility.risk_flags
    if eligibility.refund_amount > auto_approval_limit:
        risk_flags = risk_flags or (RetailActionRisk.FINANCIAL,)
    request = ReturnRequest(
        return_id=f"RET-{uuid4().hex[:8].upper()}",
        order_id=order_id,
        item_id=item_id,
        reason=reason,
        status="pending_review" if risk_flags else "approved",
        idempotency_key=idempotency_key,
    )
    return repository.create(request), eligibility.model_copy(update={"risk_flags": risk_flags})


class RetailApprovalService:
    """Apply a reviewer decision to a previously evaluated retail proposal."""

    def __init__(self, returns: ReturnRepository, refunds: RefundRepository) -> None:
        self.returns = returns
        self.refunds = refunds

    def apply(self, *, action: dict[str, object], kind: str) -> dict[str, object] | None:
        if action.get("action_type") != "retail_return":
            return None
        details = action.get("details")
        if not isinstance(details, dict):
            raise ValueError("retail action details are required")
        eligibility = details.get("eligibility")
        if not isinstance(eligibility, dict) or not eligibility.get("eligible"):
            raise ValueError("ineligible return cannot be approved")
        order_id = str(details.get("order_id"))
        item_id = str(details.get("item_id"))
        reason = str(details.get("reason"))
        idempotency_key = str(details.get("idempotency_key"))
        existing, _ = request_return(
            self.returns,
            order_id=order_id,
            item_id=item_id,
            reason=reason,
            idempotency_key=idempotency_key,
            auto_approval_limit=Decimal("0"),
        )
        if existing is None:
            raise ValueError("return proposal could not be persisted")
        if kind in {"reject_refund", "reject"}:
            rejected = self.returns.transition(existing.return_id, "rejected", expected_version=1)
            self._audit("return_rejected", existing.return_id, {"review_action": kind})
            return rejected.model_dump(mode="json")
        if kind not in {"approve_return", "approve_refund", "approve"}:
            return existing.model_dump(mode="json")
        approved = self.returns.transition(existing.return_id, "approved", expected_version=1)
        if kind != "approve_refund":
            self._audit("return_approved", approved.return_id, {"review_action": kind})
            return approved.model_dump(mode="json")
        amount = Decimal(str(eligibility.get("refund_amount", "0")))
        proposal = self.refunds.propose(approved.return_id, amount)
        result = self.refunds.transition(str(proposal["refund_id"]), "approved", expected_version=1)
        self._audit("refund_approved", str(result["refund_id"]), {"review_action": kind, "amount": str(result["amount"])})
        return result

    def _audit(self, event_type: str, entity_id: str, details: dict[str, object]) -> None:
        with self.returns.database.connect() as c:
            c.execute(
                "INSERT INTO retail_audit_events (event_type,entity_id,details_json,created_at) VALUES (?,?,?,?)",
                (event_type, entity_id, json.dumps(details), datetime.now(UTC).isoformat()),
            )
