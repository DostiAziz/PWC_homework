from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from pwc_support.agents.contracts import (
    ConversationMemory,
    RoutingTask,
    SpecialistName,
    SpecialistResult,
    SpecialistStatus,
    SupportIntent,
)
from pwc_support.agents.order import (
    OrderAgentState,
    OrderNextStep,
    OrderPlanner,
    OrderToolbox,
    build_order_agent,
)


def _memory(phase: str = "new", attempts: int = 0) -> ConversationMemory:
    return ConversationMemory(
        conversation_id=uuid4(),
        client_id="CUS-1001",
        active_specialist=SpecialistName.ORDER,
        active_task_id="order-1",
        state_json={"phase": phase, "lookup_attempts": attempts},
        version=0,
        updated_at=datetime.now(UTC),
    )


class FakeOrderTools(OrderToolbox):
    def __init__(self) -> None:
        self.lookup_call_count = 0
        self.inspect_call_count = 0
        self.cancel_call_count = 0

    def lookup_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None:
        self.lookup_call_count += 1
        if order_id == "ORD-1001":
            return {
                "order_id": order_id,
                "customer_id": customer_id,
                "status": "processing",
                "payment_status": "paid",
                "fulfilment_status": "processing",
                "total": "129.00",
                "currency": "EUR",
                "shipped_at": None,
                "version": 1,
                "items": [],
            }
        return None

    def inspect_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None:
        self.inspect_call_count += 1
        return self.lookup_order(order_id, customer_id)


class FakeOrderPlanner(OrderPlanner):
    def next_step(self, state: OrderAgentState) -> OrderNextStep:
        pass


@pytest.fixture
def fake_order_tools() -> FakeOrderTools:
    return FakeOrderTools()


class AgentResultWrapper:
    def __init__(self, result: SpecialistResult, tools: FakeOrderTools, memory_update: ConversationMemory | None):
        self._result = result
        self.tools = tools
        self.memory_update = memory_update

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)

def invoke_order_agent(request: str, memory: ConversationMemory) -> AgentResultWrapper:
    tools = FakeOrderTools()
    planner = FakeOrderPlanner()
    graph = build_order_agent(tools=tools, planner=planner, max_steps=8)
    
    import re
    match = re.search(r"ORD-[A-Z0-9-]+", request)
    entities = {"order_id": match.group(0)} if match else {}
    
    task = RoutingTask(
        task_id="order-1",
        intent=SupportIntent.ORDER_CANCELLATION,
        specialist=SpecialistName.ORDER,
        user_text=request,
        entities=entities,
        depends_on=(),
    )
    
    initial_state = {
        "task": task,
        "memory": memory,
        "customer_id": memory.client_id,
        "request": request,
        "order_id": task.entities.get("order_id"),
        "phase": memory.state_json.get("phase", "new"),
        "lookup_attempts": memory.state_json.get("lookup_attempts", 0),
        "messages": [],
    }
    
    final_state = graph.invoke(initial_state)
    result = final_state.get("specialist_result")
    memory_update = final_state.get("memory_update")
    return AgentResultWrapper(result, tools, memory_update)


def test_missing_order_id_requests_it_without_lookup() -> None:
    result = invoke_order_agent("I want to cancel my order", memory=_memory())
    assert result.status is SpecialistStatus.NEEDS_INFORMATION
    assert result.customer_message == "Please provide your order number so I can check it."
    assert result.tools.lookup_call_count == 0


def test_first_lookup_miss_requests_correction_and_persists_attempt() -> None:
    result = invoke_order_agent("ORD-9999", memory=_memory(phase="awaiting_order_id"))
    assert result.status is SpecialistStatus.CORRECTION_REQUESTED
    assert result.memory_update.state_json["lookup_attempts"] == 1


def test_second_lookup_miss_returns_not_found_without_review() -> None:
    result = invoke_order_agent("ORD-8888", memory=_memory(phase="awaiting_correction", attempts=1))
    assert result.status is SpecialistStatus.ORDER_NOT_FOUND
    assert result.review_packet is None


def test_found_cancellable_order_returns_review_packet_without_mutation() -> None:
    result = invoke_order_agent("Please cancel ORD-1001", memory=_memory())
    assert result.status is SpecialistStatus.REVIEW_REQUIRED
    assert result.review_packet is not None
    assert result.review_packet.order.status == "processing"
    assert getattr(result.tools, "cancel_call_count", 0) == 0
