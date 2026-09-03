from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from pwc_support.domain.models import ReturnEligibility, ReturnRequest, RetailActionRisk
from pwc_support.storage.retail_repositories import ReturnRepository


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
