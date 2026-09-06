from typing import Any

from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from llm.ollama import OllamaUnavailable
from services.chat import AgentService


class StreamingFakeGraph:
    def __init__(self) -> None:
        self.invocations: list[Any] = []

    def stream(
        self, payload: Any, config: Any = None, stream_mode: list[str] | None = None
    ) -> Any:
        self.invocations.append((payload, config))
        if isinstance(payload, Command):
            yield ("updates", {"confirm": {"steps": ["cancel_order"]}})
            yield (
                "messages",
                (AIMessageChunk(content="Order cancelled."), {"langgraph_node": "agent"}),
            )
            yield ("updates", {"respond": {"response": "Order cancelled.", "citations": ()}})
            return

        text = payload["messages"][-1]["content"]
        if "cancel" in text:
            class _I:
                def __init__(self) -> None:
                    self.value = {
                        "order_id": "ORD-2001",
                        "summary": "Cancel order ORD-2001 for 79.99 EUR",
                    }
            yield ("updates", {"__interrupt__": [_I()]})
            return

        if "tools" in text:
            yield ("updates", {"tools": {"steps": ["search_knowledge_base"]}})
            yield ("messages", (AIMessageChunk(content="Found "), {"langgraph_node": "agent"}))
            yield ("messages", (AIMessageChunk(content="policy."), {"langgraph_node": "agent"}))
            yield ("updates", {"respond": {"response": "Found policy.", "citations": ()}})
            return

        yield ("messages", (AIMessageChunk(content="Hello "), {"langgraph_node": "agent"}))
        yield ("messages", (AIMessageChunk(content="world!"), {"langgraph_node": "agent"}))
        yield ("updates", {"respond": {"response": "Hello world!", "citations": ()}})


def test_stream_submit_yields_tokens_and_done() -> None:
    service = AgentService(StreamingFakeGraph())
    events = list(service.stream_submit(body="hi", customer_id="CUS-1001", thread_id="t1"))

    token_events = [e for e in events if e.kind == "token"]
    assert "".join(e.token for e in token_events) == "Hello world!"

    done_events = [e for e in events if e.kind == "done"]
    assert len(done_events) == 1
    assert done_events[0].reply is not None
    assert done_events[0].reply.message == "Hello world!"
    assert done_events[0].reply.status == "answered"


def test_stream_submit_yields_tool_steps() -> None:
    service = AgentService(StreamingFakeGraph())
    events = list(service.stream_submit(body="use tools", customer_id="CUS-1001", thread_id="t2"))

    step_events = [e for e in events if e.kind == "step"]
    assert len(step_events) == 1
    assert step_events[0].step == "search_knowledge_base"

    token_events = [e for e in events if e.kind == "token"]
    assert "".join(e.token for e in token_events) == "Found policy."

    done_events = [e for e in events if e.kind == "done"]
    assert done_events[0].reply is not None
    assert done_events[0].reply.steps == ("search_knowledge_base",)


def test_stream_submit_handles_interrupt() -> None:
    service = AgentService(StreamingFakeGraph())
    events = list(
        service.stream_submit(body="cancel ORD-2001", customer_id="CUS-1001", thread_id="t3")
    )

    interrupt_events = [e for e in events if e.kind == "interrupt"]
    assert len(interrupt_events) == 1
    reply = interrupt_events[0].reply
    assert reply is not None
    assert reply.awaiting_confirmation is True
    assert "Cancel order ORD-2001" in reply.preview


def test_stream_resume_completes_confirmation() -> None:
    service = AgentService(StreamingFakeGraph())
    list(service.stream_submit(body="cancel ORD-2001", customer_id="CUS-1001", thread_id="t4"))
    resume_events = list(
        service.stream_resume(thread_id="t4", customer_id="CUS-1001", decision="yes")
    )

    step_events = [e for e in resume_events if e.kind == "step"]
    assert any(e.step == "cancel_order" for e in step_events)

    done_events = [e for e in resume_events if e.kind == "done"]
    assert len(done_events) == 1
    assert done_events[0].reply is not None
    assert done_events[0].reply.message == "Order cancelled."


def test_stream_submit_access_denied_for_different_customer() -> None:
    service = AgentService(StreamingFakeGraph())
    list(service.stream_submit(body="hi", customer_id="CUS-1001", thread_id="shared_thread"))
    events = list(
        service.stream_submit(body="hi", customer_id="CUS-9999", thread_id="shared_thread")
    )

    assert len(events) == 1
    assert events[0].kind == "error"
    assert events[0].reply is not None
    assert events[0].reply.status == "invalid_request"
    assert "Access denied" in events[0].reply.message


def test_stream_handles_ollama_unavailable() -> None:
    class FailingGraph:
        def stream(self, *args: Any, **kwargs: Any) -> Any:
            raise OllamaUnavailable("Ollama is offline")

    service = AgentService(FailingGraph())
    events = list(service.stream_submit(body="hi", customer_id="CUS-1001", thread_id="t5"))

    assert len(events) == 1
    assert events[0].kind == "error"
    assert events[0].reply is not None
    assert events[0].reply.status == "unavailable"
    assert "ensure Ollama is running" in events[0].reply.message
