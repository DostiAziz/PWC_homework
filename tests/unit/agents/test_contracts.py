from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from pwc_support.agents.contracts import (
    CancellationReviewPacket,
    OrderInvestigation,
    PolicyFinding,
    RoutingTask,
    SpecialistName,
    SupportIntent,
)


def test_routing_task_accepts_multiple_specialist_intents() -> None:
    task = RoutingTask(
        task_id="order-1",
        intent=SupportIntent.ORDER_CANCELLATION,
        specialist=SpecialistName.ORDER,
        user_text="Cancel order ORD-1001 if it has not shipped",
        entities={"order_id": "ORD-1001"},
        depends_on=(),
    )
    assert task.specialist is SpecialistName.ORDER


def test_cancellation_review_packet_requires_verified_order_facts() -> None:
    packet = CancellationReviewPacket(
        conversation_id=uuid4(),
        client_id="CUS-1001",
        original_request="Cancel my order",
        order=OrderInvestigation(
            order_id="ORD-1001",
            customer_id="CUS-1001",
            status="processing",
            total=Decimal("129.00"),
            currency="EUR",
            payment_status="paid",
            fulfilment_status="processing",
            shipped_at=None,
            version=1,
            items=(),
        ),
        policy_findings=(
            PolicyFinding(rule_id="not_shipped", passed=True, message="Order has not shipped"),
        ),
        recommended_action="human_review",
    )
    assert packet.order.order_id == "ORD-1001"
