from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from pwc_support.config import Settings
from pwc_support.domain.models import (
    OrderSummary,
    ProductSummary,
    ProposedAction,
    RetailActionRisk,
    RetailIntent,
    ReturnEligibility,
    ReturnRequest,
    ReviewDecisionKind,
)
from pwc_support.domain.state import SupportState


def test_product_summary_accepts_exact_retail_facts() -> None:
    product = ProductSummary(
        product_id="PROD-1001",
        name="Trail Shell",
        category="jackets",
        price=Decimal("129.99"),
        currency="EUR",
        stock=7,
        attributes={"waterproof": True, "colour": "navy"},
        active_offer={"offer_id": "OFFER-10", "discount_percent": 10},
    )

    assert product.price == Decimal("129.99")
    assert product.currency == "EUR"
    assert product.stock == 7


def test_order_summary_accepts_items_and_delivery_timestamp() -> None:
    delivered_at = datetime(2026, 8, 20, 10, 30, tzinfo=UTC)
    order = OrderSummary(
        order_id="ORD-1001",
        customer_id="CUS-1001",
        status="delivered",
        total=Decimal("129.99"),
        currency="EUR",
        items=({"item_id": "ITEM-1", "product_id": "PROD-1001", "quantity": 1},),
        delivered_at=delivered_at,
    )

    assert order.total == Decimal("129.99")
    assert order.delivered_at == delivered_at


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            ProductSummary,
            {
                "product_id": "PROD-1",
                "name": "Invalid",
                "category": "jackets",
                "price": Decimal("-1.00"),
                "currency": "EUR",
                "stock": 1,
                "attributes": {},
                "active_offer": None,
            },
        ),
        (
            ReturnEligibility,
            {
                "eligible": False,
                "reason": "Outside return window",
                "deadline": None,
                "refund_amount": Decimal("-0.01"),
                "risk_flags": (RetailActionRisk.POLICY_EXCEPTION,),
            },
        ),
    ],
)
def test_retail_models_reject_negative_money(
    model: type[object], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model(**payload)


def test_retail_models_reject_invalid_currency_and_unbounded_ids() -> None:
    common = {
        "name": "Trail Shell",
        "category": "jackets",
        "price": Decimal("10.00"),
        "stock": 1,
        "attributes": {},
        "active_offer": None,
    }

    with pytest.raises(ValidationError):
        ProductSummary(product_id="PROD-1", currency="euro", **common)
    with pytest.raises(ValidationError):
        ProductSummary(product_id="X" * 65, currency="EUR", **common)


def test_retail_enums_reject_unknown_values() -> None:
    with pytest.raises(ValueError):
        RetailIntent("browse_everything")
    with pytest.raises(ValueError):
        RetailActionRisk("unlimited_refund")


def test_return_request_and_eligibility_are_typed() -> None:
    request = ReturnRequest(
        return_id="RET-1001",
        order_id="ORD-1001",
        item_id="ITEM-1",
        reason="Wrong size",
        status="pending",
        idempotency_key="chat-message-1001",
    )
    eligibility = ReturnEligibility(
        eligible=True,
        reason="Within return window",
        deadline=datetime(2026, 9, 19, tzinfo=UTC),
        refund_amount=Decimal("129.99"),
        risk_flags=(RetailActionRisk.FINANCIAL,),
    )

    assert request.status == "pending"
    assert eligibility.risk_flags == (RetailActionRisk.FINANCIAL,)


def test_settings_expose_retail_policy_configuration(tmp_path: object) -> None:
    settings = Settings(data_dir=tmp_path)

    assert settings.retail_db_path == settings.data_dir / "state" / "retail.sqlite3"
    assert settings.refund_auto_approval_limit == Decimal("100.00")
    assert settings.return_window_days == 30


def test_support_state_declares_retail_workflow_slices() -> None:
    annotations = SupportState.__annotations__

    assert {"retail_intent", "retail_answer", "action_proposal"} <= set(annotations)


def test_support_state_declares_specialist_routing_slices() -> None:
    annotations = SupportState.__annotations__

    assert {
        "routing_decision",
        "routing_tasks",
        "active_specialist",
        "conversation_memory",
        "specialist_results",
        "joined_response_parts",
    } <= set(annotations)


def test_review_decision_kind_covers_cancellation_and_return_outcomes() -> None:
    assert ReviewDecisionKind.APPROVE_CANCELLATION == "approve_cancellation"
    assert ReviewDecisionKind.REJECT_CANCELLATION == "reject_cancellation"
    assert ReviewDecisionKind.APPROVE_RETURN == "approve_return"
    assert ReviewDecisionKind.REJECT_RETURN == "reject_return"
    assert ReviewDecisionKind.APPROVE_REFUND == "approve_refund"
    assert ReviewDecisionKind.REJECT_REFUND == "reject_refund"


def test_proposed_action_accepts_retail_cancellation() -> None:
    action = ProposedAction(
        action_type="retail_cancellation",
        description="Cancel order ORD-1001 pending review",
    )

    assert action.action_type == "retail_cancellation"
