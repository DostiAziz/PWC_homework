from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from pwc_support.domain.models import RetailActionRisk, ReturnEligibility, ReturnRequest
from pwc_support.storage.retail_repositories import RefundRepository, ReturnRepository, OrderRepository


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
    eligibility = evaluate_return(
        repository, order_id=order_id, item_id=item_id, reason=reason, now=now
    )
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

    def __init__(self, returns: ReturnRepository, refunds: RefundRepository, orders: OrderRepository | None = None) -> None:
        self.returns = returns
        self.refunds = refunds
        self.orders = orders

    def apply(self, *, action: dict[str, object], kind: str, review_id: str | None = None) -> dict[str, object] | None:
        action_type = action.get("action_type")
        if action_type not in ("retail_return", "retail_cancellation"):
            return None

        details = action.get("details")
        if not isinstance(details, dict):
            raise ValueError("retail action details are required")

        if action_type == "retail_cancellation":
            if self.orders is None:
                raise RuntimeError("OrderRepository is required for cancellation")
            if review_id is None:
                raise ValueError("review_id is required for cancellation")
            
            order = details.get("order")
            if not isinstance(order, dict):
                raise ValueError("order details required")
            
            order_id = str(order.get("order_id"))
            customer_id = str(order.get("customer_id"))
            version = int(str(order.get("version", 0)))
            
            if kind == "reject_cancellation":
                self._audit("cancellation_rejected", order_id, {"review_action": kind})
                return {"status": "rejected"}
                
            if kind == "approve_cancellation":
                result = self.orders.cancel(
                    order_id, 
                    customer_id, 
                    expected_version=version, 
                    review_id=review_id
                )
                self._audit("cancellation_approved", order_id, {"review_action": kind})
                return {"status": result.status, "replayed": result.replayed}
                
            return None
        investigation = details.get("investigation")
        if investigation is not None:
            if not investigation.get("eligible"):
                raise ValueError("ineligible return cannot be approved")
            order_id = str(investigation.get("order_id"))
            item_id = str(investigation.get("item_id"))
            reason = str(details.get("original_request", "Return"))
            # use review_id for idempotency if not provided
            idempotency_key = str(details.get("idempotency_key", review_id))
            refund_amount = Decimal(str(investigation.get("unit_price", "0"))) * int(str(investigation.get("quantity", "1")))
        else:
            eligibility = details.get("eligibility")
            if not isinstance(eligibility, dict) or not eligibility.get("eligible"):
                raise ValueError("ineligible return cannot be approved")
            order_id = str(details.get("order_id"))
            item_id = str(details.get("item_id"))
            reason = str(details.get("reason"))
            idempotency_key = str(details.get("idempotency_key", review_id))
            refund_amount = Decimal(str(eligibility.get("refund_amount", "0")))

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
        if kind in {"reject_refund", "reject_return", "reject"}:
            if existing.status == "rejected":
                return existing.model_dump(mode="json")
            if existing.status != "pending_review":
                raise ValueError("return version conflict")
            rejected = self.returns.transition(existing.return_id, "rejected", expected_version=1)
            self._audit("return_rejected", existing.return_id, {"review_action": kind})
            return rejected.model_dump(mode="json")
        
        if kind not in {"approve_return", "approve_refund", "approve"}:
            return existing.model_dump(mode="json")
            
        if existing.status == "approved":
            approved = existing
        else:
            if existing.status != "pending_review":
                raise ValueError("return version conflict")
            approved = self.returns.transition(existing.return_id, "approved", expected_version=1)
            if kind != "approve_refund":
                self._audit("return_approved", approved.return_id, {"review_action": kind})
            
        if kind != "approve_refund":
            return approved.model_dump(mode="json")
            
        proposal = self.refunds.propose(approved.return_id, refund_amount)
        if proposal["status"] == "approved":
            return proposal
        result = self.refunds.transition(str(proposal["refund_id"]), "approved", expected_version=1)
        self._audit(
            "refund_approved",
            str(result["refund_id"]),
            {"review_action": kind, "amount": str(result["amount"])},
        )
        return result

    def _audit(self, event_type: str, entity_id: str, details: dict[str, object]) -> None:
        db = self.returns.database if self.returns else self.orders.database
        with db.connect() as c:
            c.execute(
                "INSERT INTO retail_audit_events ("
                "event_type,entity_id,details_json,created_at) VALUES (?,?,?,?)",
                (event_type, entity_id, json.dumps(details), datetime.now(UTC).isoformat()),
            )
