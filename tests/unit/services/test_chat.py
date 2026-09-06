from typing import Any

from langgraph.types import Command

from services.chat import AgentService


class FakeGraph:
    def __init__(self) -> None:
        self.invocations: list[Any] = []

    def invoke(self, payload, config=None):
        self.invocations.append((payload, config))
        if isinstance(payload, Command):
            return {
                "response": "Order ORD-2001 has been cancelled.",
                "citations": (),
                "steps": ["cancel_order"],
            }
        text = payload["messages"][-1]["content"]
        if "cancel" in text:

            class _I:
                def __init__(self) -> None:
                    self.value = {
                        "order_id": "ORD-2001",
                        "summary": "Cancel order ORD-2001 for 79.99 EUR",
                    }

            return {"__interrupt__": [_I()]}
        return {
            "response": "Order ORD-5001 is shipped.",
            "citations": (),
            "steps": ["get_order_status"],
        }


def test_submit_projects_answer() -> None:
    reply = AgentService(FakeGraph()).submit(
        thread_id="t", body="where is ORD-5001?", customer_id="CUS-1001"
    )
    assert reply.message == "Order ORD-5001 is shipped."
    assert reply.awaiting_confirmation is False
    assert reply.steps == ("get_order_status",)


def test_submit_surfaces_confirmation_interrupt() -> None:
    reply = AgentService(FakeGraph()).submit(
        thread_id="t", body="cancel ORD-2001", customer_id="CUS-1001"
    )
    assert reply.awaiting_confirmation is True
    assert "Cancel order ORD-2001" in reply.preview


def test_resume_projects_final_answer() -> None:
    graph = FakeGraph()
    service = AgentService(graph)
    service.submit(thread_id="t", body="cancel ORD-2001", customer_id="CUS-1001")
    reply = service.resume(thread_id="t", customer_id="CUS-1001", decision="yes")
    assert "cancelled" in reply.message.lower()
    assert isinstance(graph.invocations[-1][0], Command)


def test_thread_ownership_rejects_different_customer() -> None:
    service = AgentService(FakeGraph())
    reply1 = service.submit(thread_id="t", body="where is ORD-5001?", customer_id="CUS-1001")
    assert reply1.status == "answered"

    reply2 = service.submit(thread_id="t", body="hello", customer_id="CUS-9999")
    assert reply2.status == "invalid_request"
    assert "Access denied" in reply2.message


def test_resume_rejects_different_or_unregistered_customer() -> None:
    service = AgentService(FakeGraph())
    service.submit(thread_id="t", body="cancel ORD-2001", customer_id="CUS-1001")

    denied = service.resume(thread_id="t", customer_id="CUS-9999", decision="yes")
    assert denied.status == "invalid_request"
    assert "Access denied" in denied.message

    expired = service.resume(thread_id="unknown-thread", customer_id="CUS-1001", decision="yes")
    assert expired.status == "invalid_request"
    assert "expired or is invalid" in expired.message


def test_submit_while_awaiting_confirmation_rejects_with_notice() -> None:
    service = AgentService(FakeGraph())
    service.submit(thread_id="t", body="cancel ORD-2001", customer_id="CUS-1001")
    collision = service.submit(thread_id="t", body="another question", customer_id="CUS-1001")
    assert collision.awaiting_confirmation is True
    assert collision.status == "awaiting_confirmation"
    assert "already pending" in collision.message


def test_submit_handles_graph_failure() -> None:
    class Boom:
        def invoke(self, *_a, **_k):
            raise RuntimeError("offline")

    reply = AgentService(Boom()).submit(thread_id="t", body="hi", customer_id="CUS-1001")
    assert "went wrong" in reply.message or "unavailable" in reply.message
    assert reply.status == "unavailable"

